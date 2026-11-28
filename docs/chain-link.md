# Chain card links

[← Back to the README](../README.md)

A chain measured on its own gets a card whose QR code opens the phone page set up for that chain: its pitch and half width as measured with calipers, and its colours. The card holds a link; this page is its grammar. The decision behind it is D30 in [web-build.md](web-build.md) (work queue item 14, *The link*).

The link is built and read by `splinewire/chainlink.py` and its TypeScript port `web/src/pipeline/chainlink.ts`. Both are checked against one file of test vectors, `web/src/pipeline/chainlink-vectors.json` (good links, refused links with their messages, built links), so the two give the same answer and the same words. `splinewire chain-link` makes a chain's link and card.

## The link

```
https://micahflock.github.io/spline-wire/phone/#cal=1;base=snap;p=12.987;h=4.39;k=k
```

One fragment parameter, `cal`, beside any others in the fragment (`#pair=…&cal=…` and `#cal=…&pair=…` both work; each reader strips only its own key). Being in the fragment, it never reaches a server. Its value is `;`-separated fields, the version first and the check character last:

```
value    = version *( ";" field ) ";k=" check
version  = "1"
field    = name "=" text          ; name: lowercase letters
check    = one of 0-9 a-z
```

| Field | Meaning | Values | If left out |
|---|---|---|---|
| `1` | the format version (first, no name) | `1` | required |
| `base` | the chain's preset, `data/chain-<base>.yaml` (`oring` is `data/chain.yaml`) | a preset this site has | required |
| `p` | pitch, mm | digits with up to 3 decimals; within ±2% of the preset's pitch | required |
| `h` | half width (pin line to what touches the target), mm | digits with up to 3 decimals; within ±10% of the preset's | the preset's |
| `shade` | the fiducials' shade against the links | `light`, `dark` | the preset's |
| `ch` | the colour channel detection reads (the spec's `gray_channel`, work queue item 13) | `luma`, `r`, `g`, `b` | the preset's (`luma`) |
| `n` | pins in the chain | a whole number, 3 to 200, no leading zero | the preset's |
| `k` | the check character (last) | below | required |

Everything else about the chain (the fiducial's kind and size, the contact model, the bend limit) is the preset's design and comes from `base`. **A preset named on a printed card is never changed in place**: a revised design takes a new name, since cards are permanent and the site is not. The ±2% band is around the named preset's pitch, so a run with another nominal pitch names its own preset.

Reading a link:

- The value is percent-decoded (a QR reader or a messenger may have written `;` as `%3B`), trimmed and case-folded, so `1;BASE=SNAP;P=12.987;…` reads the same as the lowercase link. A link may be given whole, as its fragment (`#cal=…`), as `cal=…` or as the value alone.
- Checked in this order, the first failure is the message: the value starts with a version number; a version above 1 is from a newer site ("reload the page"); the last field is `k=` and one character (else the link was cut short); the check character; each field is `name=value`, given once, and known (version 1 refuses a field it does not know); `base` and `p` are given; the preset exists; the numbers' form; the bands; the words.
- Fields between the version and `k` may come in any order. The builder writes them in the order of the table, numbers rounded half up to 3 decimals and written without trailing zeros (`13`, not `13.000`), and leaves out what is not given.
- Errors are sentences a person can act on, for example "This chain card's code is damaged (check character): scan it again or type the pitch in Settings." (Python: `ChainLinkError`, a `ValueError`; TypeScript: `ChainLinkError`, a `PipelineError`; each has a `code`, which the vectors name.)

## The check character

Luhn mod N with N = 36 over the alphabet `0123456789abcdefghijklmnopqrstuvwxyz`, computed on the value before `;k=`, counting only its digits and letters (case-folded); `;`, `=` and `.` are skipped. The QR code corrects its own damage; the check is for links typed by hand or cut short.

To make it: give each counted character its place in the alphabet (`0`–`9` are 0–9, `a` is 10, … `z` is 35). From the right, double every other one, starting with the rightmost. Of each doubled number keep `a // 36 + a % 36` (its two base-36 digits summed). Add everything up; the check character is the one at place `(36 − sum % 36) % 36`.

To check: append the check character and do the same with the doubling starting one place to the left (the check character itself is not doubled); the sum is a multiple of 36.

What it catches (measured in `tests/test_chainlink.py` and `chainlink.test.ts`): every change of one counted character (665 of 665 on the example below) and every swap of two neighbouring counted characters except `0`↔`z` (1 pair of 630). It does not see a moved decimal point (`.` is not counted), but a moved point puts the pitch or the half width far outside its band, which is refused.

### Worked example

`1;base=snap;p=13` counts `1 b a s e s n a p p 1 3`:

| Character | Place | Weight | × | Kept |
|---|---|---|---|---|
| `1` | 1 | ×1 | 1 | 1 |
| `b` | 11 | ×2 | 22 | 22 |
| `a` | 10 | ×1 | 10 | 10 |
| `s` | 28 | ×2 | 56 | 21 (1·36 + 20: 1 + 20) |
| `e` | 14 | ×1 | 14 | 14 |
| `s` | 28 | ×2 | 56 | 21 |
| `n` | 23 | ×1 | 23 | 23 |
| `a` | 10 | ×2 | 20 | 20 |
| `p` | 25 | ×1 | 25 | 25 |
| `p` | 25 | ×2 | 50 | 15 (1 + 14) |
| `1` | 1 | ×1 | 1 | 1 |
| `3` | 3 | ×2 | 6 | 6 |

The sum is 179; 179 % 36 = 35; (36 − 35) % 36 = 1, so the check character is `1` and the link is `1;base=snap;p=13;k=1`.

The same algorithm over the alphabet `abcdef` gives `e` for `abcdef`, the example in Wikipedia's article on the Luhn mod N algorithm; it is the first of the vectors' check characters.

### More examples

| Chain | `cal` value |
|---|---|
| snap-fit, pitch 13 mm | `1;base=snap;p=13;k=1` |
| snap-fit, pitch 12.987, half width 4.39 | `1;base=snap;p=12.987;h=4.39;k=k` |
| snap-fit, pitch 12.98749 (rounded) | `1;base=snap;p=12.987;k=u` |
| snap-fit, 13.2, 4.4, dark fiducials, blue channel, 40 pins | `1;base=snap;p=13.2;h=4.4;shade=dark;ch=b;n=40;k=g` |
| O-ring chain, pitch 9.95, half width 4 | `1;base=oring;p=9.95;h=4;k=1` |

## Making a card

```bash
uv run splinewire chain-link --pitch 12.987 --half-width 4.39 --qr out/card.svg
# https://micahflock.github.io/spline-wire/phone/?debug#cal=1;base=snap;p=12.987;h=4.39;k=k
```

`--shade dark`, `--channel b`, `--pins 20` and `--base` add the other fields; `--page` sets the phone page (the default carries `?debug` while chain cards are being tested, and loses it once the feature is promoted). The card is 85.6 × 54 mm (a bank card): the QR code (error correction M, with its 4-module quiet zone), the numbers in words and the link in small type for a person to type. Print it at 100% scale.

To measure with the numbers on a computer: `splinewire measure PHOTO --chain-link URL`, or `--pitch` and `--half-width` over the chain file's. `splinewire synth --pitch P` renders a chain off its nominal pitch, to test both.
