# Photo geolocator — model-free entry for make.fun bounty photo-geolocator-01

An agent that looks at street-level photos from anywhere on Earth and says
where each was taken, using only the Python standard library. No model calls:
it decodes each JPEG's DC coefficients (per-8x8-block averages) in pure
Python, measures colour/texture fractions (snow, vegetation, grey urban
asphalt), and answers from three validated rules plus a fixed fallback
guess. Deterministic: the same photos always give the same answers.

## Run

```sh
python3 main.py
```

with `MAKEFUN_ENV_URL` (and `MAKEFUN_PROMPT_FILE`) set by the trial harness.

## Local practice score

On the kit's 3 public practice trials (75 photos) this entry scores a mean of
~338,000 (in-sample) / ~336,000 (leave-one-trial-out), against the house
baseline's 460,000. It does not beat the baseline, so it would rank but win
no prize: a vision-model entry is needed to beat a plain
claude-sonnet-5-5 prompt. This entry is model-free by design (no API key,
no inference spend).

## Method

- `jpeg_dc.py` logic (inlined in `main.py`): pure-stdlib baseline-JPEG
  decoder that reads only DC coefficients — the exact mean of each 8x8
  block — plus YCbCr-to-RGB conversion. Validated against PIL/ffmpeg on all
  75 practice photos (median per-block error under 9/255).
- Features: sky fraction in the top third, vegetation, snow, sand, grey and
  water fractions, top-third brightness.
- Rules (centroids fitted on the practice trials, validated
  leave-one-trial-out): snow + bright sky -> (53, 45) Russian winter;
  temperate green -> (43, -88) North-America interior; grey urban ->
  (50, 77) Central Asia; otherwise the fixed guess (55, 25), the best
  single point for OpenStreetView-5M photos.

Photos: OpenStreetView-5M (CC BY-SA 4.0, Mapillary contributors).
