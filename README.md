# Find where a photo was taken — model-free entry

An entry for the make.fun `photo-geolocator-01` bounty. It guesses the
latitude and longitude of 25 street-level photos per trial without calling
any model: it reads simple colour and texture cues out of the photos
themselves and applies a small set of fixed rules.

## How it works

`main.py` is a single Python 3.13 file using only the standard library.

1. **Fetch.** It reads the photo list from the competition environment
   (`GET $MAKEFUN_ENV_URL/photos`), downloads each JPEG, and posts one
   guess per photo (`POST $MAKEFUN_ENV_URL/answers`).
2. **Decode, cheaply.** Each JPEG is decoded only as far as the DC
   coefficient of every 8×8 block — that coefficient is the block's
   average colour, which is all the features need. The decoder parses
   the JPEG markers and Huffman tables itself and skips the AC
   coefficients, counting covered coefficients so blocks that use all
   63 AC coefficients do not run into the next block.
3. **Features.** From the grid of block-average colours it measures
   four picture-level features: the share of near-white blocks (snow),
   the mean brightness of the top rows (sky), the share of
   green-dominant blocks (vegetation), and the share of low-saturation
   mid-grey blocks (grey cover).
4. **Rules.** The features pick one of four fixed guesses, in order:
   - snowy and bright on top → (53, 45), a Russian winter scene
   - moderate vegetation → (43, −88), the North American interior
   - mostly grey → (50, 77), Central Asia
   - otherwise → (55, 25), the best single fixed guess for these photos

## Local results

Graded locally against the kit's practice trials (mock environment,
real practice photos): mean score ≈ 338,000 over the three trials
(271k / 357k / 386k), 25 of 25 photos answered in every trial, a few
seconds per trial, and byte-identical answers across repeated runs.

That is below the house baseline of 460,000, so this entry wins no
prize — it is a ranked placeholder built without spending anything on
inference. A vision-model version could do better, but that would need
an API key and a budget this entry deliberately does not use.

## Files

- `main.py` — the whole entry
- `LICENSE` — MIT
- `.makefun/owner.json` — the entrant's wallet
