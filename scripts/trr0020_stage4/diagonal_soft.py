"""Full-vocabulary inversion with per-position gradients and shared forward context."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0017"))
from discrete_parallel import DiscreteParallel
from reset_soft import ResetSoftVocabulary

class DiagonalSoftVocabulary(ResetSoftVocabulary):
    def forward(self,z,pe,mask):
        return DiscreteParallel.diagonal_forward(self,z,pe,mask)
