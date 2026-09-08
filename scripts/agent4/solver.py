"""Bounded SipIt-inspired inversion. No truth or inverse-model input."""
from dataclasses import dataclass, asdict
import time
import torch

@dataclass(frozen=True)
class Settings:
    steps: int = 128
    learning_rate: float = 1.0
    snap_every: int = 50
    atol: float = 1e-5
    rtol: float = 1e-5
    token_seconds: float = 30.0
    record_seconds: float = 600.0
    seed: int = 4401


def continuous_forward(prefix, hidden):
    """Same unpadded contiguous geometry as forward_full; autograd enabled."""
    batch, tokens, _ = hidden.shape
    positions = torch.arange(tokens, device=hidden.device).view(1,-1).expand(batch,-1)
    rotary = prefix.rotary_emb(hidden, positions)
    mask = prefix._causal_mask(hidden, start_pos=0, total_tokens=tokens)
    for layer in prefix.layers:
        hidden = prefix._hidden(layer(hidden, attention_mask=mask,
            position_ids=positions, use_cache=False, position_embeddings=rotary))
    return hidden


def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


def reconstruct(prefix, observation, settings=Settings(), bos=128000, guard=None):
    if any(p.requires_grad for p in prefix.parameters()):
        raise ValueError('prefix weights must be frozen')
    if observation.ndim != 2 or not torch.isfinite(observation).all():
        raise ValueError('expected finite [tokens, hidden] observations')
    table = prefix.embed_tokens.weight.detach()
    device = table.device
    # FP32 scoring preserves raw magnitudes. No normalization of forward inputs.
    scoring = table.float()
    norms = scoring.square().sum(-1)
    generator = torch.Generator(device='cpu').manual_seed(settings.seed)
    tokens, traces = [bos], []
    sync(device); start = time.perf_counter()
    for pos in range(1, len(observation)):
        if time.perf_counter()-start > settings.record_seconds:
            tokens.extend([-1]*(len(observation)-pos)); break
        if guard: guard()
        sync(device); token_start = time.perf_counter()
        target = observation[pos].to(device).float()
        committed = table[torch.tensor(tokens,device=device)].detach().unsqueeze(0)
        initial = int(torch.randint(len(table),(1,),generator=generator).item())
        z = table[initial].float().clone().requires_grad_()
        candidate = initial
        tried = torch.zeros(len(table),device=device,dtype=torch.bool)
        checks, gradient_norms, losses = [], [], []
        best_id, best_loss, accepted = initial, float('inf'), False
        steps = 0; scans = 0; snaps = 0; reason = 'budget_exhausted'
        for step in range(settings.steps):
            if step and time.perf_counter()-token_start > settings.token_seconds:
                reason='token_timeout'; break
            # Verify the current candidate, then optimize/propose the next one.
            ids = torch.tensor([tokens+[candidate]],device=device)
            with torch.no_grad():
                predicted = prefix.forward_full(ids)[0,-1].float()
                residual = float((predicted-target).square().mean())
            if not torch.isfinite(predicted).all(): raise RuntimeError('nonfinite forward')
            checks.append({'token':candidate,'mse':residual})
            tried[candidate] = True
            if residual < best_loss:
                best_id, best_loss = candidate, residual
            if torch.allclose(predicted,target,atol=settings.atol,rtol=settings.rtol):
                accepted=True; reason='residual_accept'; break
            hidden = torch.cat([committed,z.to(table.dtype).view(1,1,-1)],dim=1)
            prediction = continuous_forward(prefix,hidden)[0,-1].float()
            loss=(prediction-target).square().mean()
            gradient, = torch.autograd.grad(loss,z)
            if not torch.isfinite(gradient).all() or not torch.isfinite(loss):
                raise RuntimeError('nonfinite gradient/loss')
            gradient_norms.append(float(gradient.norm())); losses.append(float(loss.detach()))
            steps += 1
            with torch.no_grad():
                gradient = gradient / gradient.norm().clamp(min=1)
                lr = settings.learning_rate * max(.01,1-.99*step/settings.snap_every)
                z -= lr*gradient
                distances=norms-2*(scoring@z)+z.square().sum()
                distances.masked_fill_(tried,float('inf'))
                candidate=int(distances.argmin()); scans+=1
                if (step+1)%settings.snap_every==0:
                    z.copy_(table[candidate]); snaps+=1
        tokens.append(best_id)
        sync(device)
        traces.append({'position':pos,'initial_token':initial,'token':best_id,
            'best_mse':best_loss,'accepted':accepted,'reason':reason,
            'gradient_steps':steps,'vocabulary_scans':scans,
            'vocabulary_rows_scored':scans*len(table),'snaps':snaps,'restarts':0,
            'checks':checks,'gradient_norms':gradient_norms,'continuous_losses':losses,
            'seconds':time.perf_counter()-token_start})
    sync(device)
    return {'tokens':tokens,'trace':traces,'seconds':time.perf_counter()-start,
        'settings':asdict(settings),'learned_proposer':False,'cache':'none; full own-prefix recomputation'}
