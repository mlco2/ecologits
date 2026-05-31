import json
import os

from ecologits.electricity_mix_repository import electricity_mixes
from ecologits.impacts.video import compute_video_impacts
from ecologits.log import logger
from ecologits.status_messages import ModelNotRegisteredError, ZoneNotRegisteredError
from ecologits.tracers.utils import ImpactsOutput
from ecologits.utils.range_value import RangeValue


_NAMED_RESOLUTIONS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


_VIDEO_MODELS_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "data", "video_models.json"
)
with open(_VIDEO_MODELS_PATH) as _fd:
    _video_models_data = json.load(_fd)
_HARDWARE_CONFIGURATIONS = _video_models_data["hardware_configurations"]
_PROVIDER_CONFIGURATIONS = _video_models_data["provider_configurations"]
_MODELS_INFO = {
    m["model_name"]: m
    for m in _video_models_data["models"]
}


def video_impacts(
        model_name: str,
        resolution: str,
        duration: float,
        with_audio: bool = True,
        datacenter_location: str | None = "WOR",
        datacenter_pue: float | RangeValue | None = None,
        datacenter_wue: float | RangeValue | None = None,
) -> ImpactsOutput:
    """
    Determines the impacts of generating a video based on the specified model, resolution,
    and duration. Calculates and returns the detailed impacts related to computational
    resources, energy consumption, and data processing associated with the generation task.

    Arguments:
        model_name: The name of the model selected for video generation.
        resolution: The resolution of the video to be generated, typically provided
            as a string (e.g., "1920x1080" or "1080p").
        duration: The length of the video to generate, specified in seconds.
        with_audio: Whether the video generation also includes audio.
        datacenter_location: ISO 3166-1 alpha-3 code of the datacenter location (WOR by default).
        datacenter_pue: Power Usage Effectiveness of the datacenter. Uses the provider default when omitted.
        datacenter_wue: Water Usage Effectiveness of the datacenter in L/kWh. Uses the provider default when omitted.

    Returns:
        An ImpactsOutput object containing details of the computed impacts.
    """
    model_info = _MODELS_INFO.get(model_name)
    if model_info is None:
        error = ModelNotRegisteredError(message=f"Could not find video model `{model_name}`.")
        logger.warning_once(str(error))
        return ImpactsOutput(errors=[error])

    provider_configuration = _PROVIDER_CONFIGURATIONS[model_info["provider"]]
    if datacenter_location is None:
        datacenter_location = provider_configuration["datacenter_location"]
    if datacenter_pue is None:
        datacenter_pue = parse_value_or_range(provider_configuration["datacenter_pue"])
    if datacenter_wue is None:
        datacenter_wue = parse_value_or_range(provider_configuration["datacenter_wue"])

    if_electricity_mix = electricity_mixes.find_electricity_mix(zone=datacenter_location)
    if if_electricity_mix is None:
        error = ZoneNotRegisteredError(
            message=f"Could not find electricity mix for `{datacenter_location}` zone."
        )
        logger.warning_once(str(error))
        return ImpactsOutput(errors=[error])

    width, height = parse_resolution(resolution)
    frames_count = duration_to_frames(duration)

    hardware = _HARDWARE_CONFIGURATIONS[model_info["hardware"]]
    server_power = hardware["server_power"]
    server_embodied = hardware["server_embodied"]
    accelerator_embodied = hardware["accelerator_embodied"]
    server_accelerator_count = hardware["number_of_accelerators"]

    # Server power in the data is in W; the dag expects kW.
    server_accelerator_power: float | RangeValue = RangeValue(
        min=server_power["p2_5"] / 1000,
        max=server_power["p97_5"] / 1000,
    )

    # When generating with audio, the regression baseline already includes audio,
    # so apply no scaling. When generating without audio, use the model's calibrated
    # multiplier (1.0 for models without native audio support).
    regression = dict(model_info["regression_parameters"])
    if with_audio:
        regression["non_audio_weight"] = 1.0

    impacts = compute_video_impacts(
        video_width=width,
        video_height=height,
        video_frames_count=frames_count,
        server_accelerator_power=server_accelerator_power,
        if_electricity_mix_adpe=if_electricity_mix.adpe,
        if_electricity_mix_pe=if_electricity_mix.pe,
        if_electricity_mix_gwp=if_electricity_mix.gwp,
        if_electricity_mix_wue=if_electricity_mix.wue,
        datacenter_pue=datacenter_pue,
        datacenter_wue=datacenter_wue,
        accelerator_embodied_gwp=accelerator_embodied["gwp"],
        accelerator_embodied_adpe=accelerator_embodied["adpe"],
        accelerator_embodied_pe=accelerator_embodied["pe"],
        server_embodied_gwp=server_embodied["gwp"],
        server_embodied_adpe=server_embodied["adpe"],
        server_embodied_pe=server_embodied["pe"],
        server_accelerator_count=server_accelerator_count,
        **regression,
    )
    impacts = ImpactsOutput.model_validate(impacts.model_dump())

    if if_electricity_mix.has_warnings:
        for warning in if_electricity_mix.warnings:
            logger.warning_once(str(warning))
            impacts.add_warning(warning)

    return impacts


def parse_resolution(resolution: str) -> tuple[int, int]:
    """
    Parses a resolution string and returns its width and height as a tuple.

    This function supports named resolutions defined in a private dictionary and
    direct width x height formats (e.g., "1920x1080"). For named resolutions,
    the function will return the associated width and height values. If the input
    does not match these formats, an error is raised.

    Arguments:
        resolution: str
            The resolution string to parse. It may be a named resolution or in the
            format "widthxheight".

    Returns:
        tuple[int, int]
            A tuple containing the width and height, respectively.

    Raises:
        ValueError: Raised if the resolution is unsupported or formatted incorrectly.
    """
    key = resolution.strip().lower()
    if key in _NAMED_RESOLUTIONS:
        return _NAMED_RESOLUTIONS[key]
    if "x" in key:
        width_str, height_str = key.split("x", 1)
        return int(width_str), int(height_str)
    raise ValueError(f"Unsupported resolution: {resolution!r}")


def parse_value_or_range(value: float | dict[str, float]) -> float | RangeValue:
    """
    Parses a fixed numeric value or min/max range from model data.

    Arguments:
        value: Either a fixed numeric value or a dict containing "min" and "max" values.

    Returns:
        The fixed value or a RangeValue.
    """
    if isinstance(value, dict):
        return RangeValue(min=value["min"], max=value["max"])
    return value


def duration_to_frames(duration: float) -> int:
    """
    Converts a video duration in seconds to a frame count at 24 fps.

    Arguments:
        duration: The length of the video in seconds.

    Returns:
        The number of frames as an integer.
    """
    return int(duration * 24 + 1)
