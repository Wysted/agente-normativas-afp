"""Stable lock keys of Postgres advisory locks."""

from agintis.modules.knowledge.infrastructure.postgres.locks import (
    session_lock_key,
    xact_lock_keys,
)
from agintis.shared_kernel.ids import TenantId

T = TenantId("tenant")
INT64 = 2**63
INT32 = 2**31


def test_keys_are_stable_across_releases() -> None:
    # Pinned: a change would let two workers on different versions lock the same source apart.
    assert session_lock_key(T, "p", "file:///a.pdf") == 6977138126276225834
    assert xact_lock_keys(T, "p", "file:///a.pdf") == (2060433724, 868782137)


def test_session_key_is_a_signed_64_bit_integer() -> None:
    keys = {session_lock_key(T, "p", f"s{i}") for i in range(200)}
    assert all(-INT64 <= key < INT64 for key in keys)
    assert any(key < 0 for key in keys)  # signed range is really used
    assert len(keys) == 200


def test_xact_keys_are_signed_32_bit_pairs() -> None:
    for i in range(50):
        high, low = xact_lock_keys(T, "p", f"s{i}")
        assert -INT32 <= high < INT32
        assert -INT32 <= low < INT32


def test_every_part_of_the_identity_matters_and_boundaries_are_unambiguous() -> None:
    base = session_lock_key(T, "p", "s")
    assert session_lock_key(TenantId("other"), "p", "s") != base
    assert session_lock_key(T, "q", "s") != base
    assert session_lock_key(T, "p", "t") != base
    # "ab"+"c" vs "a"+"bc": the NUL separator keeps them apart.
    assert session_lock_key(TenantId("ab"), "c", "s") != session_lock_key(TenantId("a"), "bc", "s")


def test_lone_surrogates_in_a_uri_do_not_crash() -> None:
    assert isinstance(session_lock_key(T, "p", "file:///\udcff.pdf"), int)
