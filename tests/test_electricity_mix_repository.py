from ecologits.electricity_mix_repository import ElectricityMix, ElectricityMixRepository
from ecologits.status_messages import (
    ElectricityMixADPeDefaultWarning,
    ElectricityMixPEDefaultWarning,
    ElectricityMixWUEDefaultWarning,
)


def test_create_electricity_mix_repository_default() -> None:
    electricity_mixes = ElectricityMixRepository.from_json()
    assert isinstance(electricity_mixes, ElectricityMixRepository)
    assert electricity_mixes.find_electricity_mix(zone="BEL") is not None


def test_create_electricity_mix_repository_from_scratch() -> None:
    electricity_mixes = ElectricityMixRepository([
        ElectricityMix(
            zone="wonderland",
            adpe=0.,
            pe=0.,
            gwp=0.,
            wue=0.
        )
    ])
    assert electricity_mixes.find_electricity_mix(zone="wonderland") is not None


def test_find_unknown_zone() -> None:
    electricity_mixes = ElectricityMixRepository.from_json()
    assert electricity_mixes.find_electricity_mix(zone="AAA") is None


def test_list_electricity_mixes() -> None:
    em1 = ElectricityMix(zone="AAA", adpe=0., pe=0., gwp=0., wue=0.)
    em2 = ElectricityMix(zone="BBB", adpe=1., pe=1., gwp=1., wue=1.)
    repository = ElectricityMixRepository([em1, em2])
    electricity_mixes = repository.list_electricity_mixes()
    assert len(electricity_mixes) == len([em1, em2])
    assert em1 in electricity_mixes
    assert em2 in electricity_mixes


def test_electricity_mix_warnings_from_json() -> None:
    electricity_mixes = ElectricityMixRepository.from_json()
    electricity_mix = electricity_mixes.find_electricity_mix(zone="ABW")

    assert electricity_mix is not None
    assert electricity_mix.has_warnings
    assert isinstance(electricity_mix.warnings[0], ElectricityMixADPeDefaultWarning)
    assert isinstance(electricity_mix.warnings[1], ElectricityMixPEDefaultWarning)
    assert isinstance(electricity_mix.warnings[2], ElectricityMixWUEDefaultWarning)


def test_electricity_mix_null_warnings_from_json() -> None:
    electricity_mixes = ElectricityMixRepository.from_json()
    electricity_mix = electricity_mixes.find_electricity_mix(zone="ARG")

    assert electricity_mix is not None
    assert not electricity_mix.has_warnings
    assert electricity_mix.warnings == []
