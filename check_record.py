#!/usr/bin/env python3
"""check_record.py — report the physical structure of a .pqd file's records.

For diagnosing a "Source file integrity: INCOMPLETE" report without sending
the file anywhere. Everything printed is structural — record offsets, sizes,
tags and links — and none of it is a measurement or a customer identity, so
the output is safe to paste into an email or a ticket.

    python3 check_record.py "C:/Pronto4w/Exported/SITE 7-20-2026.pqd"

The question it answers: when a record will not decode, is the file short of
what its own headers declare (an interrupted export or copy), or did the
writer emit a record header with no body at all (a conformance problem in
whatever wrote the file)?
"""

import sys
import zlib
from pathlib import Path

import pqdif

#: The smallest zlib stream that exists — deflate of nothing at all. Under
#: record-level compression IEEE 1159.3-2019 clause 6.2 requires the header to
#: carry the *compressed* size, so any body smaller than this was never a
#: compressed stream, whatever the header claims.
MIN_ZLIB_STREAM = len(zlib.compress(b""))


def main(path: Path) -> int:
    size = path.stat().st_size
    f = pqdif.PQDIFFile(path)
    records = f.records

    print(f"file            {path.name}")
    print(f"size on disk    {size:,} bytes")
    print(f"records         {len(records)}")
    print(f"compression     {'record-level' if f.compressed else 'none'}")
    print()

    # A record is worth reporting either because its header does not add up or
    # because the body it points at will not decode. The two are different
    # failures and the file's owner needs to know which one they have.
    suspect = []
    for r in records:
        why = None
        if r.body_size < MIN_ZLIB_STREAM:
            why = "header declares a body too short to be a compressed stream"
        elif r.missing_bytes:
            why = "file is shorter than this header declares"
        elif r.next_header_intact is False:
            why = "no record header where this one links to"
        else:
            try:
                r.body(f.compressed)
            except Exception as exc:
                why = f"body did not decode ({type(exc).__name__})"
        if why:
            suspect.append((r, why))

    if not suspect:
        print("Every record header is self-consistent, every body is at least "
              "as long as a zlib stream can be, and every body decoded. "
              "Nothing structural to report.")
        return 0

    print(f"{len(suspect)} record(s) worth looking at:")
    print()
    for r, why in suspect:
        print(f"  ── {why}")
        end = r.position + r.header_size + r.body_size
        print(f"  offset            {r.position:,}")
        print(f"  tag               {r.tag}")
        print(f"  declared body     {r.body_size:,} bytes")
        print(f"  bytes present     {len(r.raw_body):,}")
        print(f"  to next record    {len(r.span_body):,}")
        print(f"  links to          {r.next_position:,}"
              + ("   (0 = writer marked this the last record)"
                 if r.next_position == 0 else ""))
        print(f"  next header found {r.next_header_intact}")
        print(f"  missing bytes     {r.missing_bytes:,}")
        print(f"  header ends at    {r.position + r.header_size:,}"
              + ("   (= end of file: nothing was written after the header)"
                 if r.position + r.header_size == size else ""))
        print(f"  body would end at {end:,}"
              + ("   (past end of file)" if end > size else ""))

        # The two diagnoses this script exists to separate.
        if r.body_size == 0:
            print("  -> The header declares no body at all. A compressed body "
                  f"cannot be shorter than {MIN_ZLIB_STREAM} bytes, and clause "
                  "4.2.2 requires a body to begin with a collection element, "
                  "so this record was written without one. That is the writer's "
                  "doing, not a transfer that lost bytes.")
        elif r.missing_bytes:
            print(f"  -> The file is {r.missing_bytes:,} bytes shorter than "
                  "this header declares. Bytes that were written are not here, "
                  "which is an interrupted export or copy. Re-export.")
        print()
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    target = Path(sys.argv[1])
    if not target.exists():
        print(f"No such file: {target}")
        sys.exit(2)
    sys.exit(main(target))
