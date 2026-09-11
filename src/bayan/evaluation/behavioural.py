"""Lab 6: behavioural test generators/runners."""


def _predict_one(predict_fn, text):
    prediction = predict_fn([text])

    if isinstance(prediction, (list, tuple)):
        return prediction[0]

    return prediction


def run_behavioural_suite(
    predict_fn,
    *,
    invariance_tests=None,
    directional_tests=None,
    mft_tests=None,
):
    """
    Run Bayan behavioural tests.

    Supported test types:
    - invariance:
        prediction should stay the same after a harmless change.
    - directional:
        prediction should change in the expected direction.
    - MFT:
        prediction should match the required expected output.
    """

    invariance_tests = invariance_tests or []
    directional_tests = directional_tests or []
    mft_tests = mft_tests or []

    # -----------------------------------------------------
    # Invariance tests
    # -----------------------------------------------------

    invariance_passed = 0

    for test in invariance_tests:
        original = test.get(
            "original",
            test.get("base"),
        )

        transformed = test.get(
            "transformed",
            test.get("variant"),
        )

        if original is None or transformed is None:
            raise ValueError(
                "Invariance tests require "
                "'original'/'base' and "
                "'transformed'/'variant'."
            )

        original_prediction = _predict_one(
            predict_fn,
            original,
        )

        transformed_prediction = _predict_one(
            predict_fn,
            transformed,
        )

        if (
            original_prediction
            == transformed_prediction
        ):
            invariance_passed += 1

    invariance_total = len(
        invariance_tests
    )

    if invariance_total:
        invariance_rate = (
            invariance_passed
            / invariance_total
        )
    else:
        invariance_rate = 0.0

    # -----------------------------------------------------
    # Directional tests
    # -----------------------------------------------------

    directional_passed = 0

    for test in directional_tests:
        original = test.get(
            "original",
            test.get("base"),
        )

        transformed = test.get(
            "transformed",
            test.get("variant"),
        )

        expected = test.get(
            "expected",
        )

        if original is None or transformed is None:
            raise ValueError(
                "Directional tests require "
                "'original'/'base' and "
                "'transformed'/'variant'."
            )

        original_prediction = _predict_one(
            predict_fn,
            original,
        )

        transformed_prediction = _predict_one(
            predict_fn,
            transformed,
        )

        # If the skeleton gives an expected result,
        # check against it directly.
        if expected is not None:
            passed = (
                transformed_prediction
                == expected
            )

        # Otherwise, the prediction must change
        # in the intended direction.
        else:
            direction = test.get(
                "direction",
                "change",
            )

            if direction == "increase":
                passed = (
                    transformed_prediction
                    > original_prediction
                )

            elif direction == "decrease":
                passed = (
                    transformed_prediction
                    < original_prediction
                )

            else:
                passed = (
                    transformed_prediction
                    != original_prediction
                )

        if passed:
            directional_passed += 1

    directional_total = len(
        directional_tests
    )

    if directional_total:
        directional_rate = (
            directional_passed
            / directional_total
        )
    else:
        directional_rate = 0.0

    # -----------------------------------------------------
    # Minimum-functionality tests
    # -----------------------------------------------------

    mft_passed = 0

    for test in mft_tests:
        text = test.get(
            "text",
            test.get("input"),
        )

        expected = test.get(
            "expected",
            test.get("label"),
        )

        if text is None or expected is None:
            raise ValueError(
                "MFT tests require "
                "'text'/'input' and "
                "'expected'/'label'."
            )

        prediction = _predict_one(
            predict_fn,
            text,
        )

        if prediction == expected:
            mft_passed += 1

    mft_total = len(
        mft_tests
    )

    if mft_total:
        mft_rate = (
            mft_passed
            / mft_total
        )
    else:
        mft_rate = 0.0

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    return {
        "invariance": {
            "passed": invariance_passed,
            "total": invariance_total,
            "pass_rate": invariance_rate,
        },
        "directional": {
            "passed": directional_passed,
            "total": directional_total,
            "pass_rate": directional_rate,
        },
        "mft": {
            "passed": mft_passed,
            "total": mft_total,
            "pass_rate": mft_rate,
        },
    }