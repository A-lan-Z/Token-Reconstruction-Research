"""Normalize the observed-output equation without restricting vocabulary."""
import torch
def equation(prediction,target,matvec,rmatvec):
    norm=prediction.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    unit=prediction/norm
    target_unit=target/target.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    def tangent(value):
        response=matvec(value)
        return (response-unit*(unit*response).sum(-1,keepdim=True))/norm
    def adjoint(value):
        projected=(value-unit*(unit*value).sum(-1,keepdim=True))/norm
        return rmatvec(projected)
    return target_unit-unit,tangent,adjoint
