import pytest

from underwriting.enrichment.providers import (
    FinancialProvider,
    IdentityProvider,
    MedicalHistoryProvider,
    PrescriptionProvider,
)


@pytest.mark.parametrize(
    "provider_class",
    [
        IdentityProvider,
        MedicalHistoryProvider,
        PrescriptionProvider,
        FinancialProvider,
    ],
)
def test_provider_interfaces_cannot_be_instantiated_directly(
    provider_class,
):
    with pytest.raises(TypeError):
        provider_class()