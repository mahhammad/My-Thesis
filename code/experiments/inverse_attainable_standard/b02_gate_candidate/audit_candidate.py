"""B02 candidate V2: append-only preflight attempts, no graph and no admission.

Trusted output root must be outside checkout, owned by the executing user and
not writable by group/other; it must be pre-provisioned or securely created
under an already-private parent. No output root => DENY, never claim audit.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import os
import stat
import sys
import time
import uuid
from pure_gate import (TrustedSpec, GateError, read_pinned_bundle,
                       verify_b02_inputs, SNAPSHOT_COMMIT, PLAN_COMMIT,
                       CH4_COMMIT)


def _private_audit_directory(directory: Path, protected_dirs=()):
    """Open trusted directory by FD. Reject symlink and publicly writable paths."""
    directory = Path(directory).absolute()
    for protected in protected_dirs:
        check = Path(protected).absolute()
        if directory == check or directory.is_relative_to(check):
            raise GateError("E_AUDIT_ROOT_UNAVAILABLE", "AUDIT_ROOT",
                            "output within historical/source tree")
    if directory.is_symlink() or directory.parent.is_symlink():
        raise GateError("E_AUDIT_ROOT_UNAVAILABLE", "AUDIT_ROOT", "symlink audit directory/parent")
    try:
        parent = directory.parent.resolve(strict=True)
        parent_stat = parent.stat()
        if (parent_stat.st_uid != os.getuid() or parent_stat.st_mode & 0o022
                or not stat.S_ISDIR(parent_stat.st_mode)):
            raise GateError("E_AUDIT_ROOT_UNAVAILABLE", "AUDIT_ROOT",
                            "parent not private/owned")
        try:
            os.mkdir(directory, 0o700)
        except FileExistsError:
            pass
        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(directory, flags)
        actual = os.fstat(fd)
        named = directory.stat(follow_symlinks=False)
        if (not stat.S_ISDIR(actual.st_mode) or actual.st_uid != os.getuid()
                or actual.st_mode & 0o077 or (actual.st_dev, actual.st_ino) !=
                (named.st_dev, named.st_ino)):
            os.close(fd)
            raise GateError("E_AUDIT_ROOT_UNAVAILABLE", "AUDIT_ROOT",
                            "audit directory not private, owned, and stable")
        return directory, fd
    except GateError:
        raise
    except (OSError, ValueError) as exc:
        raise GateError("E_AUDIT_ROOT_UNAVAILABLE", "AUDIT_ROOT",
                        type(exc).__name__) from exc


def write_record(directory: Path, record: dict, *, protected_dirs=()):
    """One independent immutable JSON record; O_EXCL; fsync file and directory.

    Returns only after a stable trusted root and completed durable write.
    """
    directory, dfd = _private_audit_directory(Path(directory), protected_dirs)
    try:
        attempt = record.get("attempt_id")
        if not isinstance(attempt, str) or len(attempt) != 32 or any(
                c not in "0123456789abcdef" for c in attempt):
            attempt = uuid.uuid4().hex
        phase = record.get("phase", "terminal")
        if phase not in ("start", "terminal"):
            raise GateError("E_AUDIT_PHASE", "AUDIT", "unsupported phase")
        body = (json.dumps(dict(record, attempt_id=attempt),
                           sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
        destname = attempt + "." + phase + ".json"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os,"O_NOFOLLOW",0)
        fd = os.open(destname, flags, 0o600, dir_fd=dfd)
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(body)
                out.flush()
                os.fsync(out.fileno())
            os.fsync(dfd)
        except BaseException:
            # Never overwrite or recycle partially written evidence.
            raise
        named = directory.stat(follow_symlinks=False)
        live = os.fstat(dfd)
        if (named.st_ino, named.st_dev) != (live.st_ino, live.st_dev):
            raise GateError("E_AUDIT_ROOT_UNAVAILABLE", "AUDIT_ROOT",
                            "root moved during write")
        return directory / destname
    except GateError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise GateError("E_AUDIT_WRITE", "AUDIT", type(exc).__name__) from exc
    finally:
        os.close(dfd)


def _base_record(attempt, scope, trust_digest=None):
    return {"attempt_id": attempt, "scope": scope,
            "trusted_manifest_sha256": trust_digest, "version": "B02_GATE_CANDIDATE_V2",
            "graph_constructed": 0, "optimizer_steps": 0}


def _reject(record, exc, started):
    if isinstance(exc, GateError):
        record.update(status="DENIED" if exc.code == "DENY_NO_AUTHORIZATION"
                      else "REJECTED", code=exc.code,
                      stage=exc.stage, reason=exc.explanation)
    else:
        record.update(status="REJECTED", code="E_UNEXPECTED",
                      stage="AUDIT", reason=type(exc).__name__)
    record.update(phase="terminal", wall_seconds=time.monotonic()-started)
    return record


def audit(bundle, trust, scope, outdir, *, protected_dirs=()):
    """For already decoded inputs; v1-compatible one-record explicit denial."""
    started = time.monotonic()
    record = _base_record(uuid.uuid4().hex, scope,
                          getattr(trust, "manifest_sha256", None))
    try:
        verify_b02_inputs(bundle=bundle, trusted_spec=trust, requested_scope=scope)
    except BaseException as exc:
        _reject(record, exc, started)
    else:
        record.update(status="REJECTED", code="E_FAIL_OPEN", stage="AUDIT",
                      phase="terminal", wall_seconds=time.monotonic()-started)
    path = write_record(Path(outdir), record, protected_dirs=protected_dirs)
    return path, record


def _external_trust(trust_path, snapshot_dir, repo_root):
    if trust_path is None:
        raise GateError("E_TRUST_MISSING", "PREFLIGHT_TRUST", "trust path not supplied")
    candidate = Path(trust_path).absolute()
    if candidate.is_symlink() or candidate.parent.is_symlink():
        raise GateError("E_TRUST_UNSAFE", "PREFLIGHT_TRUST", "symlink trust file")
    try:
        trusted = candidate.resolve(strict=True)
        if not trusted.is_file():
            raise GateError("E_TRUST_UNSAFE", "PREFLIGHT_TRUST", "not a regular trust file")
        for root in (snapshot_dir, repo_root):
            if root is not None and trusted.is_relative_to(Path(root).resolve(strict=False)):
                raise GateError("E_TRUST_UNSAFE", "PREFLIGHT_TRUST", "trust inside checkout")
        raw = trusted.read_bytes()
    except FileNotFoundError as exc:
        raise GateError("E_TRUST_MISSING", "PREFLIGHT_TRUST") from exc
    except GateError:
        raise
    except OSError as exc:
        raise GateError("E_TRUST_IO", "PREFLIGHT_TRUST", type(exc).__name__) from exc
    try:
        obj = json.loads(raw, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
        if not isinstance(obj, dict):
            raise TypeError("trust mapping required")
    except (UnicodeError, ValueError, TypeError) as exc:
        raise GateError("E_TRUST_JSON", "PREFLIGHT_TRUST", "malformed trust JSON") from exc
    try:
        trusted_spec = TrustedSpec(**obj)
        if (not isinstance(trusted_spec.manifest_sha256, str)
                or not isinstance(trusted_spec.source_sha256, dict)
                or not isinstance(trusted_spec.array_sha256, dict)
                or not isinstance(trusted_spec.approved_runtime, dict)):
            raise TypeError("required trust field types")
    except (TypeError, KeyError, ValueError) as exc:
        raise GateError("E_TRUST_SCHEMA", "PREFLIGHT_TRUST", "missing/wrong trust fields") from exc
    return trusted_spec


def audit_preflight_attempt(*, snapshot_dir, repo_root, trusted_path,
                            audit_output, scope, refs=None):
    """Audit START precedes every untrusted read/parse/path resolution.

    Caller must select private audit root outside checkout; this routine never
    creates graph, never imports a graph module and always returns DENY.
    """
    started = time.monotonic()
    attempt = uuid.uuid4().hex
    roots = tuple(Path(r).absolute() for r in (snapshot_dir, repo_root) if r is not None)
    record = _base_record(attempt, scope)
    # If this fails, no audit evidence is claimed. No admission path exists.
    startpath = write_record(audit_output, dict(record, phase="start",
                            status="STARTED", stage="PREFLIGHT"),
                            protected_dirs=roots)
    try:
        if sys.flags.optimize:
            raise GateError("E_RUNTIME_OPTIMIZED", "G19", "python -O/-OO forbidden")
        trust = _external_trust(trusted_path, snapshot_dir, repo_root)
        record["trusted_manifest_sha256"] = trust.manifest_sha256
        if snapshot_dir is None or repo_root is None:
            raise GateError("E_INPUT_MISSING", "PREFLIGHT_BUNDLE", "snapshot/repo root required")
        try:
            bundle = read_pinned_bundle(snapshot_dir, repo_root, refs=refs or {
                "snapshot": SNAPSHOT_COMMIT, "plan": PLAN_COMMIT, "chapter4": CH4_COMMIT})
        except GateError as exc:
            raise GateError(exc.code, "PREFLIGHT_BUNDLE", exc.explanation) from exc
        except FileNotFoundError as exc:
            raise GateError("E_INPUT_MISSING", "PREFLIGHT_BUNDLE", "missing input path") from exc
        except (PermissionError, OSError, ValueError) as exc:
            raise GateError("E_INPUT_LOADER", "PREFLIGHT_BUNDLE",
                            type(exc).__name__) from exc
        verify_b02_inputs(bundle=bundle, trusted_spec=trust, requested_scope=scope)
    except BaseException as exc:
        _reject(record, exc, started)
    else:
        record.update(status="REJECTED", code="E_FAIL_OPEN", stage="AUDIT",
                      phase="terminal", wall_seconds=time.monotonic()-started)
    # Keep START immutable, emit separately committed terminal evidence.
    terminal = write_record(audit_output, record, protected_dirs=roots)
    return startpath, terminal, record


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--trusted-spec", type=Path)
    parser.add_argument("--audit-output", type=Path, required=True,
                        help="pre-provisioned private root outside checkout")
    parser.add_argument("--scope", default="RESOURCE_PROBE")
    args = parser.parse_args(argv)
    try:
        started, terminal, record = audit_preflight_attempt(
            snapshot_dir=args.snapshot_dir, repo_root=args.repo_root,
            trusted_path=args.trusted_spec, audit_output=args.audit_output,
            scope=args.scope)
    except GateError as exc:
        # An unavailable output root is itself a DENY, not an audited attempt.
        print("DENY " + exc.code + " " + exc.stage, file=sys.stderr)
        return 3
    print(json.dumps({"audit_start": str(started), "audit_final": str(terminal),
                      "status": record["status"], "code": record["code"]},
                     sort_keys=True))
    # No zero exit for denied scopes; prevents accidental CI grant-by-exit.
    return 2


if __name__ == "__main__":
    sys.exit(main())
