"""Lab 4: per-model Arabic normalisation profiles."""

from dataclasses import dataclass

from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import (
    normalize_alef_ar,
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar,
)


@dataclass(frozen=True)
class ArabicProfile:
    name: str
    dediacritize: bool = False


DISPLAY_PROFILE = ArabicProfile(
    name="display",
    dediacritize=False,
)

BAYAN_AR_PROFILE = ArabicProfile(
    name="bayan_ar_v1",
    dediacritize=True,
)


def normalize_arabic(text: str, profile: ArabicProfile) -> str:
    """
    Normalize Arabic text according to the selected profile.

    display:
        Preserve the original text.

    bayan_ar_v1:
        Apply Bayan Arabic normalization for model input.
    """

    # Preserve text exactly for display.
    if profile.name == "display":
        return text

    if profile.name == "bayan_ar_v1":
        normalized = text

        # Remove Arabic Tatweel.
        normalized = normalized.replace("\u0640", "")

        # Normalize Alef forms:
        # أ إ آ -> ا
        normalized = normalize_alef_ar(normalized)

        # Normalize Alef Maksura:
        # ى -> ي
        normalized = normalize_alef_maksura_ar(normalized)

        # Normalize Teh Marbuta:
        # ة -> ه
        normalized = normalize_teh_marbuta_ar(normalized)

        # Normalize hamza-on-waw and hamza-on-ya.
        # ؤ -> و
        # ئ -> ي
        normalized = normalized.replace("ؤ", "و")
        normalized = normalized.replace("ئ", "ي")

        # Remove Arabic diacritics when requested.
        if profile.dediacritize:
            normalized = dediac_ar(normalized)

        return normalized

    raise ValueError(
        f"Unknown Arabic normalisation profile: {profile.name}"
    )


def segment(text: str) -> list[str]:
    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.tokenizers.morphological import MorphologicalTokenizer

    mle = MLEDisambiguator.pretrained()

    tokenizer = MorphologicalTokenizer(
        disambiguator=mle,
        scheme="atbtok",
        split=True,
    )

    return tokenizer.tokenize(text.split())