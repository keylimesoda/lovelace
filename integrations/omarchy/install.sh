#!/usr/bin/env bash
# Install the lovelace Omarchy widget from this checkout.
#
# `omarchy plugin add` clones a repository whose root is the plugin, so it
# cannot install this subdirectory. Omarchy does discover symlinked plugin
# folders (and `omarchy plugin remove` unlinks them), so this script links:
#
#   ~/.config/omarchy/plugins/keylimesoda.lovelace -> <checkout>/integrations/omarchy
#   ~/.local/bin/lovelace-monitor                   -> <checkout>/integrations/omarchy/lovelace-monitor
#
# `git pull` in the checkout then updates the widget; the shell reloads plugin
# code on its own.
set -euo pipefail

id=keylimesoda.lovelace
src=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
plugins_dir="$HOME/.config/omarchy/plugins"
bin_dir="$HOME/.local/bin"
state_dir="${XDG_CONFIG_HOME:-$HOME/.config}/lovelace-monitor"
old_state_dir="${XDG_CONFIG_HOME:-$HOME/.config}/ninfer-qwen"

usage() {
    cat <<'USAGE'
Usage: integrations/omarchy/install.sh [--enable] [--ninfer-qwen-alias]
       integrations/omarchy/install.sh --uninstall

  --enable              also enable the widget (manifest default: right section)
  --ninfer-qwen-alias   also link ~/.local/bin/ninfer-qwen to the helper, for
                        scripts written against the old omarchy-ninfer helper
                        (an existing regular file is kept as ninfer-qwen.bak.<time>)
  --uninstall           remove the plugin link and the helper links; keeps
                        ~/.config/lovelace-monitor (engine and vision settings)
USAGE
}

fail() {
    printf 'install.sh: %s\n' "$*" >&2
    exit 1
}

# link NAME TARGET: point NAME at TARGET, refusing to replace anything that is
# not already a symlink.
link() {
    local name=$1 target=$2
    if [[ -L $name ]]; then
        ln -sfn -- "$target" "$name"
    elif [[ -e $name ]]; then
        fail "$name exists and is not a symlink; move it away first"
    else
        ln -s -- "$target" "$name"
    fi
    printf 'linked %s -> %s\n' "$name" "$target"
}

# unlink_ours NAME: remove NAME only when it is a symlink into this checkout.
unlink_ours() {
    local name=$1
    if [[ -L $name && $(readlink -f -- "$name") == "$src"* ]]; then
        rm -f -- "$name"
        printf 'removed %s\n' "$name"
    fi
}

enable=0 alias=0 uninstall=0
for arg in "$@"; do
    case "$arg" in
        --enable) enable=1 ;;
        --ninfer-qwen-alias) alias=1 ;;
        --uninstall) uninstall=1 ;;
        -h | --help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

if (( uninstall )); then
    if [[ -L $plugins_dir/$id ]]; then
        # Disables the widget if enabled, unlinks, and rescans the shell.
        omarchy plugin remove "$id" --yes
    fi
    unlink_ours "$bin_dir/lovelace-monitor"
    unlink_ours "$bin_dir/ninfer-qwen"
    exit 0
fi

command -v omarchy >/dev/null || fail "omarchy not found; this installs an Omarchy shell plugin"
omarchy plugin validate "$src"

mkdir -p "$plugins_dir" "$bin_dir"
link "$plugins_dir/$id" "$src"
link "$bin_dir/lovelace-monitor" "$src/lovelace-monitor"

if (( alias )); then
    if [[ -f $bin_dir/ninfer-qwen && ! -L $bin_dir/ninfer-qwen ]]; then
        backup="$bin_dir/ninfer-qwen.bak.$(date +%Y%m%d%H%M%S)"
        mv -- "$bin_dir/ninfer-qwen" "$backup"
        printf 'moved the old helper to %s\n' "$backup"
    fi
    link "$bin_dir/ninfer-qwen" "$src/lovelace-monitor"
fi

# Carry the old helper's settings (engine, vision, spec) over once.
if [[ -d $old_state_dir && ! -e $state_dir ]]; then
    cp -r -- "$old_state_dir" "$state_dir"
    printf 'copied settings from %s to %s\n' "$old_state_dir" "$state_dir"
fi

# Best effort (-q): a shell that is not running picks the plugin up at start.
omarchy-shell -q shell rescanPlugins

if (( enable )); then
    # The rescan is asynchronous; wait for the shell to discover the plugin.
    for _ in $(seq 50); do
        omarchy plugin list --json | jq -e --arg id "$id" 'any(.[]; .id == $id)' >/dev/null && break
        sleep 0.1
    done
    omarchy plugin enable "$id"
else
    printf 'Enable it with: omarchy plugin enable %s [--section left|center|right]\n' "$id"
fi

if [[ -e $plugins_dir/keylimesoda.ninfer ]]; then
    printf 'note: the previous widget keylimesoda.ninfer is still installed; remove it with: omarchy plugin remove keylimesoda.ninfer\n'
fi
