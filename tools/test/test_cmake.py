from __future__ import annotations

import contextlib
import os
import typing
import unittest
import unittest.mock
from pathlib import Path

import tools.setup_helpers.cmake
import tools.setup_helpers.env  # noqa: F401 unused but resolves circular import


if typing.TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


T = typing.TypeVar("T")


class TestCMake(unittest.TestCase):
    def test_ck_sdpa_runtime_arch_guard_supports_gfx11_gfx12(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        context_h = (repo_root / "aten/src/ATen/Context.h").read_text()
        context_cpp = (repo_root / "aten/src/ATen/Context.cpp").read_text()

        self.assertIn("static bool ckSDPASupported();", context_h)

        blas_guard = context_cpp[
            context_cpp.index("bool Context::ckSupported()"):
            context_cpp.index("bool Context::ckSDPASupported()")
        ]
        sdpa_guard = context_cpp[
            context_cpp.index("bool Context::ckSDPASupported()"):
            context_cpp.index("void Context::setBlasPreferredBackend")
        ]

        self.assertIn('"gfx90a", "gfx942", "gfx950"', blas_guard)
        self.assertNotIn('"gfx11"', blas_guard)
        self.assertNotIn('"gfx12"', blas_guard)

        self.assertIn('"gfx90a", "gfx942", "gfx950", "gfx11", "gfx12"', sdpa_guard)

        rocm_fa_preference = context_cpp[
            context_cpp.index("at::ROCmFABackend Context::getROCmFAPreferredBackend()"):
            context_cpp.index(
                "CuBLASReductionOption Context::allowFP16ReductionCuBLAS()"
            )
        ]
        self.assertIn("ckSDPASupported()", rocm_fa_preference)
        self.assertNotIn("ckSupportedFlag = ckSupported()", rocm_fa_preference)

    def test_ck_sdpa_fav3_is_arch_gated(self) -> None:
        cmake = (
            Path(__file__).resolve().parents[2]
            / "aten/src/ATen/CMakeLists.txt"
        ).read_text()

        fav3_add = "add_subdirectory(native/transformers/hip/flash_attn/ck/fav_v3)"
        fav3_target = "hip_add_library(ck_sdpa_fav3 STATIC"
        fav3_link = "target_link_libraries(ck_sdpa PRIVATE ck_sdpa_fav3)"
        fav3_filter = (
            'list(FILTER CK_SDPA_FAV3_TARGETS INCLUDE REGEX "^gfx(942|950)$")'
        )
        fav3_arch_flags = (
            "list(APPEND CK_SDPA_FAV3_HIP_CLANG_FLAGS --offload-arch=${ARCH})"
        )

        self.assertIn(fav3_filter, cmake)
        self.assertIn(fav3_add, cmake)
        self.assertIn(fav3_target, cmake)
        self.assertIn(fav3_link, cmake)
        self.assertIn(fav3_arch_flags, cmake)

        gate_pos = cmake.index("set(CK_SDPA_FAV3_TARGETS")
        add_pos = cmake.index(fav3_add)
        target_pos = cmake.index(fav3_target)

        self.assertLess(gate_pos, add_pos)
        self.assertLess(add_pos, target_pos)

        ck_sdpa_sources = cmake[
            cmake.index("file(GLOB ck_sdpa_sources_hip"):
            cmake.index("set_source_files_properties(${ck_sdpa_sources_hip}")
        ]
        self.assertNotIn("fav_v3", ck_sdpa_sources)

    @unittest.mock.patch("multiprocessing.cpu_count")
    def test_build_jobs(self, mock_cpu_count: unittest.mock.MagicMock) -> None:
        """Tests that the number of build jobs comes out correctly."""
        mock_cpu_count.return_value = 13
        cases = [
            # MAX_JOBS, USE_NINJA, IS_WINDOWS,         want
            (("8", True, False), ["-j", "8"]),  # noqa: E201,E241
            ((None, True, False), None),  # noqa: E201,E241
            (("7", False, False), ["-j", "7"]),  # noqa: E201,E241
            ((None, False, False), ["-j", "13"]),  # noqa: E201,E241
            (("6", True, True), ["-j", "6"]),  # noqa: E201,E241
            ((None, True, True), None),  # noqa: E201,E241
            (("11", False, True), ["-j", "11"]),  # noqa: E201,E241
            ((None, False, True), ["-j", "13"]),  # noqa: E201,E241
        ]
        for (max_jobs, use_ninja, is_windows), want in cases:
            with self.subTest(
                MAX_JOBS=max_jobs, USE_NINJA=use_ninja, IS_WINDOWS=is_windows
            ):
                with contextlib.ExitStack() as stack:
                    stack.enter_context(env_var("MAX_JOBS", max_jobs))
                    stack.enter_context(
                        unittest.mock.patch.object(
                            tools.setup_helpers.cmake, "USE_NINJA", use_ninja
                        )
                    )
                    stack.enter_context(
                        unittest.mock.patch.object(
                            tools.setup_helpers.cmake, "IS_WINDOWS", is_windows
                        )
                    )

                    cmake = tools.setup_helpers.cmake.CMake()

                    with unittest.mock.patch.object(cmake, "run") as cmake_run:
                        cmake.build({})

                    cmake_run.assert_called_once()
                    (call,) = cmake_run.mock_calls
                    build_args, _ = call.args

                if want is None:
                    self.assertNotIn("-j", build_args)
                else:
                    self.assert_contains_sequence(build_args, want)

    @staticmethod
    def assert_contains_sequence(
        sequence: Sequence[T], subsequence: Sequence[T]
    ) -> None:
        """Raises an assertion if the subsequence is not contained in the sequence."""
        if len(subsequence) == 0:
            return  # all sequences contain the empty subsequence

        # Iterate over all windows of len(subsequence). Stop if the
        # window matches.
        for i in range(len(sequence) - len(subsequence) + 1):
            candidate = sequence[i : i + len(subsequence)]
            if len(candidate) != len(subsequence):  # sanity check
                raise AssertionError(
                    f"candidate length mismatch: {len(candidate)} != {len(subsequence)}"
                )
            if candidate == subsequence:
                return  # found it
        raise AssertionError(f"{subsequence} not found in {sequence}")


@contextlib.contextmanager
def env_var(key: str, value: str | None) -> Iterator[None]:
    """Sets/clears an environment variable within a Python context."""
    # Get the previous value and then override it.
    previous_value = os.environ.get(key)
    set_env_var(key, value)
    try:
        yield
    finally:
        # Restore to previous value.
        set_env_var(key, previous_value)


def set_env_var(key: str, value: str | None) -> None:
    """Sets/clears an environment variable."""
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
