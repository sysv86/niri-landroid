#!/usr/bin/env bash
#
# niri-landroid installer for Arch Linux and Arch-based distributions.
#
# Installs the Android 17 Easter Egg "Landroid" wallpaper for the Niri compositor and,
# if it is not present yet, the RWallpprs engine it runs on.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/sysv86/niri-landroid/main/install.sh | bash
#   ./install.sh [options]
#
# Options:
#   --prefix DIR     Install prefix for the binary (default: /usr/local)
#   --no-deps        Do not run pacman; assume every dependency is present
#   --no-start       Do not start the wallpaper after installing
#   --uninstall      Remove the installed wallpaper
#   -h, --help       Show this help

set -Eeuo pipefail

readonly SELF_URL="https://raw.githubusercontent.com/sysv86/niri-landroid/main/install.sh"
readonly REPO_URL="https://github.com/sysv86/niri-landroid.git"
readonly RWALLPPRS_INSTALL_URL="https://raw.githubusercontent.com/sysv86/rwallpprs/main/install.sh"

PREFIX="/usr/local"
INSTALL_DEPS=1
START=1
MODE="install"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

usage() { sed -n '3,20p' "$0" 2>/dev/null | sed 's/^# \{0,1\}//' || true; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)    [[ -n "${2:-}" ]] || die "--prefix needs a directory"; PREFIX="$2"; shift 2 ;;
        --prefix=*)  PREFIX="${1#*=}"; shift ;;
        --no-deps)   INSTALL_DEPS=0; shift ;;
        --no-start)  START=0; shift ;;
        --uninstall) MODE="uninstall"; shift ;;
        -h|--help)   usage; exit 0 ;;
        *)           die "unknown option: $1 (try --help)" ;;
    esac
done

# A piped script shares stdin with sudo's password prompt, which would swallow the
# rest of the script. Re-exec from a real file first.
if [[ ! -t 0 && -z "${LANDROID_INSTALL_REEXEC:-}" ]]; then
    export LANDROID_INSTALL_REEXEC=1
    tmp="$(mktemp /tmp/niri-landroid-install.XXXXXX.sh)"
    curl -fsSL "$SELF_URL" -o "$tmp" || die "could not download $SELF_URL"
    if [[ -r /dev/tty ]]; then
        exec bash "$tmp" "$@" </dev/tty
    fi
    exec bash "$tmp" "$@"
fi

[[ "$(uname -s)" == "Linux" ]] || die "this wallpaper targets Linux (Niri runs on Linux only)"

if ! command -v pacman >/dev/null 2>&1; then
    die "this script supports Arch-based distributions only (pacman not found).
Install the dependencies from README.md manually, then run: cargo build --release"
fi

if [[ "${EUID}" -ne 0 ]]; then
    command -v sudo >/dev/null 2>&1 || die "run as root or install sudo"
    SUDO="sudo"
else
    SUDO=""
fi

asroot() { $SUDO "$@"; }

if [[ "$MODE" == "uninstall" ]]; then
    log "Removing $PREFIX/bin/niri-landroid"
    asroot rm -f "$PREFIX/bin/niri-landroid"
    log "Done (the RWallpprs engine was left in place)"
    exit 0
fi

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
readonly DEPS=(
    base-devel
    rust
    clang
    pkgconf
    git
    ffmpeg
    libva
    vulkan-icd-loader
    wayland
    wayland-protocols
)

if [[ "$INSTALL_DEPS" -eq 1 ]]; then
    log "Installing build and runtime dependencies"
    asroot pacman -S --needed --noconfirm "${DEPS[@]}"
else
    log "Skipping dependency installation (--no-deps)"
fi

# ---------------------------------------------------------------------------
# The wallpaper needs the RWallpprs daemon to be displayed.
# ---------------------------------------------------------------------------
ENGINE_INSTALLED=0
if command -v rwallpprs >/dev/null 2>&1 && command -v rwallpprs-daemon >/dev/null 2>&1; then
    ENGINE_INSTALLED=1
    log "Found an existing RWallpprs installation"
else
    log "RWallpprs is not installed; installing it first"
    rw_installer="$(mktemp /tmp/rwallpprs-install.XXXXXX.sh)"
    curl -fsSL "$RWALLPPRS_INSTALL_URL" -o "$rw_installer" \
        || die "could not download the RWallpprs installer"
    # --no-deps: dependencies were already installed above.
    bash "$rw_installer" --no-deps
    rm -f "$rw_installer"
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
if [[ -f "${PWD}/Cargo.toml" && -d "${PWD}/src" ]]; then
    SRC_DIR="$PWD"
    CLEANUP=""
else
    SRC_DIR="$(mktemp -d /tmp/niri-landroid-src.XXXXXX)"
    CLEANUP="$SRC_DIR"
fi
cleanup() { [[ -n "${CLEANUP:-}" ]] && rm -rf "$CLEANUP"; }
trap cleanup EXIT

if [[ -z "$CLEANUP" ]]; then
    log "Building from the current checkout at $SRC_DIR"
else
    log "Cloning $REPO_URL"
    git clone --depth 1 "$REPO_URL" "$SRC_DIR"
fi

cd "$SRC_DIR"
log "Building niri-landroid (the first build fetches wgpu and the RWallpprs SDK)"
cargo build --release

log "Installing the wallpaper into $PREFIX/bin"
asroot install -Dm755 target/release/niri-landroid "$PREFIX/bin/niri-landroid"

# ---------------------------------------------------------------------------
# Start it on the running Niri desktop
# ---------------------------------------------------------------------------
if [[ "$START" -eq 1 ]]; then
    if [[ -n "${NIRI_SOCKET:-}" ]] || pgrep -x niri >/dev/null 2>&1; then
        log "Starting the wallpaper"
        if command -v rwallpprs >/dev/null 2>&1; then
            if ! pgrep -f rwallpprs-daemon >/dev/null 2>&1; then
                log "Starting the RWallpprs daemon"
                rwallpprs daemon --config /etc/xdg/rwallpprs/rwallpprs.toml &
                sleep 1
            fi
            if rwallpprs set-bin "$PREFIX/bin/niri-landroid" --mode animated --fps 60; then
                log "The Landroid wallpaper is running"
            else
                warn "could not set the wallpaper; start it manually with:"
                warn "  rwallpprs set-bin $PREFIX/bin/niri-landroid --mode animated --fps 60"
            fi
        fi
    else
        warn "no running Niri session detected; start the wallpaper later with:"
        warn "  rwallpprs set-bin $PREFIX/bin/niri-landroid --mode animated --fps 60"
    fi
fi

cat <<EOF

niri-landroid is installed at $PREFIX/bin/niri-landroid

Show it on the desktop:

    rwallpprs daemon --config /etc/xdg/rwallpprs/rwallpprs.toml &
    rwallpprs set-bin $PREFIX/bin/niri-landroid --mode animated --fps 60

Use a fixed universe instead of the daily seed:

    rwallpprs set-bin $PREFIX/bin/niri-landroid --mode animated --fps 60 -- --seed 12345
EOF
