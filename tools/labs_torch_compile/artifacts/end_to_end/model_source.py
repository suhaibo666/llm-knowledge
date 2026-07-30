import torch
from torch import nn

class UnifiedGraphModel(nn.Module):
    """Stable prefix used across the series; later stages progressively lower it."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))
        self.register_buffer("offset", torch.linspace(-0.2, 0.2, 4))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        activation, summary = backend_core(x, self.weight, self.offset)
        return {"activation": activation, "summary": summary}


class HigherOrderBranch(nn.Module):
    def forward(
        self, x: torch.Tensor, predicate: torch.Tensor
    ) -> torch.Tensor:
        return torch.cond(
            predicate,
            lambda value: torch.sin(value),
            lambda value: torch.cos(value),
            (x,),
        )


class InvalidHigherOrderBranch(nn.Module):
    def forward(
        self, x: torch.Tensor, predicate: torch.Tensor
    ) -> torch.Tensor:
        return torch.cond(
            predicate,
            lambda value: value.clone(),
            lambda value: value.sum(),
            (x,),
        )


def backend_core(
    x: torch.Tensor,
    weight: torch.Tensor,
    offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected = x.view(-1, 4) @ weight
    biased = projected + offset
    scratch = biased.clone()
    scratch.view(-1).add_(0.25)
    activation = torch.sin(scratch)
    return activation, activation.sum(dim=1)
