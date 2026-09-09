"""Static scientific summary of the fixed matrix; consumes scored results only."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.agent3_hybrid.common import METHODS,DOMAINS,STAGES
LABELS={'a1_k256':'Historical+A2, K256','b1_k256':'B1+A2, K256','b1_k16':'B1+A2, K16','a1_k16':'Historical+A2, K16','b1_alone':'B1 alone'}
COLORS={'a1_k256':'#475569','b1_k256':'#2563eb','b1_k16':'#059669','a1_k16':'#d97706','b1_alone':'#9333ea'}

def main(a):
    r=json.loads(Path(a.results).read_text());cells={(x['domain'],x['stage'],x['method']):x for x in r['cells']}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(2,3,figsize=(12.5,7.2),layout='constrained')
    for i,d in enumerate(DOMAINS):
        for method in METHODS:
            c=[cells[d,s,method] for s in STAGES]
            ys=([100*x['token_accuracy']['estimate'] for x in c],[100*x['exact_clips']/32 for x in c],[x['warmed_total_seconds']/32 for x in c])
            for j,y in enumerate(ys):axes[i,j].plot(STAGES,y,color=COLORS[method],marker='o',linewidth=2 if method=='b1_k16' else 1.4,linestyle='-' if method=='b1_k16' else '--',label=LABELS[method],markersize=4)
        for j,title in enumerate(('Token accuracy (%)','Exact clips (%)','Warmed seconds / clip (log scale)')):
            ax=axes[i,j];ax.set_title(d.capitalize()+' · '+title);ax.set_xticks(STAGES);ax.set_xlabel('Existing target updates');ax.grid(alpha=.2)
            if j<2:ax.set_ylim(-2,102)
            else:ax.set_yscale('log')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Static public prefix: fixed B1+A2 pilot\n32 paired fresh clips per domain; instrumented medians after warmup',fontsize=14)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('.png','.svg'):
        p=out.with_suffix(ext)
        if p.exists():raise FileExistsError(p)
        fig.savefig(p)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',required=True);main(p.parse_args())
