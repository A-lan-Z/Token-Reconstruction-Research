"""Local synthetic checks. Run: python -m unittest discover -s . -p 'test_*.py'."""
import unittest, torch
from prototype import TinyPrefix,window_search,proposals,DTYPE
from run_experiments_v1 import analytic_checks,cache_checks

class TestPrototype(unittest.TestCase):
 @classmethod
 def setUpClass(cls): torch.set_num_threads(1)
 def test_cache_and_gradient(self):
  c=cache_checks();self.assertLess(c['output_max_abs_error'],1e-12);self.assertLess(c['gradient_max_abs_error'],1e-12);self.assertTrue(c['cache_immutable'])
 def test_continuous_future_cancels(self):
  c=analytic_checks(); self.assertLess(c['linear_future_cancellation_norm'],1e-12);self.assertLess(c['projected_cross_token_jacobian_norm'],1e-12)
 def test_new_root_can_be_proposed_without_future_labels(self):
  E=torch.tensor([[-1.],[0.],[1.]],dtype=DTYPE)
  root=torch.zeros(1,1,1,dtype=DTYPE,requires_grad=True)
  current=.5*(root**3-1).square().sum()
  gl=torch.autograd.grad(current,root,retain_graph=True)[0]
  self.assertEqual(int(proposals(E,root,gl,1)[0,0,0]),1) # stays at zero
  # Every possible real-token tail is allowed; none contains hidden supervision.
  for tail in [-1.,0.,1.]:
   joint=current+.5*(2*root+tail-2).square().sum()
   gj=torch.autograd.grad(joint,root,retain_graph=True)[0]
   self.assertEqual(int(proposals(E,root,gj,1)[0,0,0]),2) # proposes +1
 def test_zero_cross_channel_supplies_no_new_gradient(self):
  z=torch.tensor([0.,0.],dtype=DTYPE,requires_grad=True)
  own=.5*(z[0]**3-1)**2
  whole=own+.5*(z[1]-1)**2
  g1=torch.autograd.grad(own,z,retain_graph=True)[0][0]
  g2=torch.autograd.grad(whole,z)[0][0]
  self.assertEqual(float(g1),float(g2))
 def test_real_tokens_and_no_labels_in_search(self):
  m=TinyPrefix(190); ids=torch.tensor([0,4,61,25])
  with torch.no_grad(): h=m.forward(m.E[ids][None])[0,1:]
  out=window_search(m,h,'joint',fast=True,block_commit=True)
  self.assertEqual(len(out['prediction']),3)
  self.assertTrue(all(isinstance(x,int) and 0<=x<128 for x in out['prediction']))
 def test_one_reverse_gradient_matches_decomposition(self):
  m=TinyPrefix(401); ids=torch.tensor([0,8,2,17,51]);cache=m.commit(ids[:2],m.empty_cache())
  z=m.E[ids[2:]][None].clone().requires_grad_(True); y=m.forward(z,cache)
  loss=y.square().sum(-1)
  parts=[torch.autograd.grad(loss[:,j].sum(),z,retain_graph=True)[0] for j in range(3)]
  full=torch.autograd.grad(loss.sum(),z)[0]
  self.assertTrue(torch.allclose(full,torch.stack(parts).sum(0),atol=1e-11,rtol=1e-11))
 def test_no_model_parameter_fitting(self):
  m=TinyPrefix(415); before=[v.clone() for l in m.layers for v in l.values()]
  ids=torch.tensor([0,8,12,15]); h=m.forward(m.E[ids][None])[0,1:].detach()
  window_search(m,h,'joint',fast=True,block_commit=True)
  self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,[v for l in m.layers for v in l.values()])))
 def test_monotone_verified_block_scores(self):
  m=TinyPrefix(401);ids=torch.tensor([0,19,37,54,12]);h=m.forward(m.E[ids][None])[0,1:].detach()
  out=window_search(m,h,'joint',fast=True,block_commit=True)
  last={}
  for e in out['trace']:
   p=e['position'];self.assertLessEqual(e['best_loss'],last.get(p,float('inf'))+1e-12);last[p]=e['best_loss']

if __name__=='__main__':unittest.main()
