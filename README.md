# Chess Assistant

On-device chess move assistant for the Chess.com iOS app. It draws suggestions as arrows on the board, either the strongest move from a bundled Stockfish 18 or a human-like move from the Maia 3 79M neural network.

Everything runs locally on the device. No network requests, and no move data leaves your phone.

## Features

Two engines to pick from. Stockfish 18 is statically linked with its NNUE nets embedded, and always goes for the strongest move. Maia 3 79M runs as a Core ML model and instead plays what a human around 1300 to 1900 ELO would play.

- ELO slider from 400 to 3500, mapped to search depth plus skill level and UCI_LimitStrength
- Up to 3 candidate arrows, each labeled with its own eval
- Centipawn or win percentage on the floating button
- Move Analysis grades every move you play from Brilliant to Blunder, tracks accuracy, and shows the full breakdown in the settings panel
- Auto Play answers for you once your opponent moves. The delay is adjustable from 0 to 5 seconds with about a second of random jitter, and roughly 10% of the time it plays the second-best move so the choices look less mechanical. In puzzles it always plays the correct solution.
- Long-press the floating ♟ button to pause or resume the assistant right away
- Live detection of about 40 openings, including the Ruy Lopez, Najdorf, Caro-Kann and Queen's Gambit, shown in the panel
- One tap copies the current FEN to your clipboard
- Accuracy stats, move grades and opening tracking reset when a game ends, detected on-device from checkmate or stalemate
- Arrow styling: color by eval or solid green, thin to thick shafts, adjustable opacity
- A ⚡ button that applies safer settings in one tap (ELO 1000, Maia on, Auto Play off)

Works in online games, bot games and puzzles.

## Install

### Jailbroken (rootless)

Add the repo in Sileo or Zebra and install:

```
https://itzzace.github.io/chess-assistant/
```

Respring after installing, then open Chess.com.

### TrollStore (no jailbreak)

Download the iOS 16 or 17 IPA from the [releases page](https://github.com/itzzace/chess-assistant/releases) and sideload it with TrollStore.

## Usage

1. Open Chess.com and start any game.
2. Tap the floating ♟ button to open the settings panel. Long-press it to pause or resume instantly.
3. Set your ELO and turn on Maia, Auto Play or Move Analysis as you like. The ⚡ button applies safer settings for you.
4. Arrows appear on the board when it's your turn. With Auto Play on, the move gets played for you.

Debug Log in the panel shows recent events such as FEN detection, engine responses and which hooks installed. FEN copies the current position.

> Warning: using an assistant breaks Chess.com's fair-play policy and can get your account banned. Anything above roughly 1500 ELO is much easier to detect.

## Building

Requires [Theos](https://theos.dev) with the iOS SDK:

```bash
# clone Stockfish and fetch its NNUE nets
git clone --depth 1 --branch sf_18 https://github.com/official-stockfish/Stockfish.git sf
cd sf/src && make net && cd ../..

# cross-compile Stockfish for iOS arm64 into a static lib
cd sf/src
SDK=$(xcrun -sdk iphoneos --show-sdk-path)
CXX=$(xcrun -sdk iphoneos -f clang++)
for f in $(find . -name '*.cpp'); do
  $CXX -arch arm64 -isysroot "$SDK" -miphoneos-version-min=15.0 \
       -std=c++17 -O3 -DNDEBUG -fno-exceptions \
       -DUSE_PTHREADS -DIS_64BIT -DUSE_POPCNT -DUSE_NEON=8 -I. -w \
       -c "$f" -o "$f.o"
done
ar rcs libstockfish.a $(find . -name '*.cpp.o' ! -name 'main.cpp.o')
cd ../..

# build the deb after exporting and validating Maia 79M on macOS
python3.12 -m venv .venv-maia
source .venv-maia/bin/activate
python -m pip install -r tools/maia79m/requirements.txt
python tools/maia79m/convert_validate.py --mode all
make package FINALPACKAGE=1
```

CI performs the same export and numerical checks on macOS before packaging. See [.github/workflows/build.yml](.github/workflows/build.yml). The official Maia 3 79M checkpoint is pinned in [tools/maia79m](tools/maia79m/README.md), converted into `tools/maia79m/maia3-79m.mlpackage`, then embedded in the tweak dylib. The app can fall back to that embedded package when the jailbreak model path is not readable.

## Project layout

| File | Purpose |
|------|---------|
| `Tweak.xm` | Main tweak: board/FEN detection, arrow rendering, settings UI, auto play |
| `engine.mm` | Stockfish embedded via UCI stream redirection (`EngineGo`, FEN legality, legal-move enumeration) |
| `maia.mm` | Maia 3 Core ML inference wrapper (79M package) |
| `web/` | Landing page / repo site |
| `Makefile` | Theos tweak build (`com.chess.assistant`, rootless) |
| `.github/workflows/` | CI: builds libstockfish.a + .deb, publishes releases & apt repo |

## Credits

- Made by @epicccccc ([YouTube](https://youtube.com/@epicccccc), Discord: `itzzace.`)
- Yousseif ([usif-x](https://github.com/usif-x)) upgraded the engine to Stockfish 18 and added Auto Play with human-like timing and puzzle solving, quick pause, opening detection, Copy FEN, auto stats reset and other tweak updates
- [Stockfish](https://stockfishchess.org), the GPL v3 chess engine
- [Maia 3 79M](https://huggingface.co/UofTCSSLab/Maia3-79M), human-like chess AI (AGPLv3)

Licensed under GPL v3 where applicable, because of the Stockfish linkage.
