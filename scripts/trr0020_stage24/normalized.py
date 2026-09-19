"""Normalize the observed-output equation without restricting vocabulary."""
import torch
def equation(prediction,target,matvec,rmatvec):
    raw_norm=prediction.norm(dim=-1,keepdim=True)
    norm=raw_norm.clamp_min(1e-12)
    active=(raw_norm>=1e-12).to(prediction.dtype)
    unit=prediction/norm
    target_unit=target/target.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    def tangent(value):
        response=matvec(value)
        return (response-active*unit*(unit*response).sum(-1,keepdim=True))/norm
    def adjoint(value):
        projected=(value-active*unit*(unit*value).sum(-1,keepdim=True))/norm
        return rmatvec(projected)
    return target_unit-unit,tangent,adjoint
