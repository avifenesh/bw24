#!/usr/bin/env bash
# Lever C grouped-vs-sequential prefill perf, N=5 interleaved per prompt shape.
set -uo pipefail

REPO=${REPO:-"$HOME/memra"}
MODEL=${MODEL:-"/data/models/step37/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"}
PROMPT4096=${PROMPT4096:-"/tmp/pipeprime-prompt-pp4096.txt"}
RAW=${RAW:-"/tmp/leverC-perf"}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
SUMMARY="$RAW/perf-summary-$TS.log"
RESULTS="$RAW/perf-results-$TS.tsv"
P512="$RAW/prompt-pp512-$TS.txt"
P2048="$RAW/prompt-pp2048-$TS.txt"
P4096="$RAW/prompt-pp4096-$TS.txt"
N=${N:-5}

mkdir -p "$RAW"
head -c 2800 "$PROMPT4096" >"$P512"
head -c 11200 "$PROMPT4096" >"$P2048"
cp -- "$PROMPT4096" "$P4096"
cd "$REPO" || exit 1

snapshot() {
  echo "snapshot $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used \
    --format=csv,noheader || true
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader || true
}

run_arm() {
  local shape=$1
  local prompt=$2
  local arm=$3
  local rep=$4
  local log="$RAW/$shape-$arm-r$rep-$TS.log"
  echo "########## $shape rep=$rep arm=$arm ##########"
  echo "raw=$log"
  snapshot
  if [[ "$arm" == grouped ]]; then
    env -u MEMRA_MOE_GROUPED -u MEMRA_MOE_STATS -u MEMRA_MOE_GATE \
      -u MEMRA_PRIME_CHUNK -u MEMRA_PRIME_PP -u MEMRA_PRIME_PIPE \
      MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 3600 \
      ./target/release/concat-prime-probe "$MODEL" ppprime \
      --prompt-a "@$prompt" --reps 1 --warmup 1 >"$log" 2>&1
  else
    env -u MEMRA_MOE_STATS -u MEMRA_MOE_GATE \
      -u MEMRA_PRIME_CHUNK -u MEMRA_PRIME_PP -u MEMRA_PRIME_PIPE \
      MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_MOE_GROUPED=0 timeout 3600 \
      ./target/release/concat-prime-probe "$MODEL" ppprime \
      --prompt-a "@$prompt" --reps 1 --warmup 1 >"$log" 2>&1
  fi
  local arm_rc=$?
  cat "$log"
  echo "$shape rep=$rep arm=$arm exit=$arm_rc"
  snapshot
  return "$arm_rc"
}

summarize_arm() {
  local shape=$1
  local arm=$2
  local times=()
  local tokens=""
  local log line parsed tok seconds
  for log in "$RAW"/"$shape"-"$arm"-r*-"$TS".log; do
    [[ -f "$log" ]] || continue
    line=$(grep -E 'ppprime MEDIAN:' "$log" | tail -1)
    [[ -n "$line" ]] || continue
    parsed=$(awk '
      /ppprime MEDIAN:/ {
        for (i = 1; i <= NF; i++) {
          if ($i == "MEDIAN:") tok = $(i + 1)
          if ($i == "in") {
            sec = $(i + 1)
            sub(/s$/, "", sec)
          }
        }
      }
      END { print tok, sec }
    ' <<<"$line")
    read -r tok seconds <<<"$parsed"
    [[ -n "$tok" && -n "$seconds" ]] || continue
    tokens=${tokens:-$tok}
    if [[ "$tokens" != "$tok" ]]; then
      echo "$shape $arm token-count mismatch: $tokens vs $tok"
      return 1
    fi
    times+=("$seconds")
  done
  if (( ${#times[@]} != N )); then
    echo "$shape $arm expected N=$N, found ${#times[@]}"
    return 1
  fi
  local sorted=()
  mapfile -t sorted < <(printf '%s\n' "${times[@]}" | sort -n)
  local median_s=${sorted[$((N / 2))]}
  local median_rate
  median_rate=$(awk -v tok="$tokens" -v sec="$median_s" \
    'BEGIN { printf "%.1f", tok / sec }')
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$shape" "$tokens" "$arm" "$N" "$median_s" "$median_rate" >>"$RESULTS"
}

report_comparison() {
  local shape=$1
  local off grouped historical delta_vs_off delta_vs_historical
  off=$(awk -F '\t' -v shape="$shape" \
    '$1 == shape && $3 == "off" { print $6 }' "$RESULTS")
  grouped=$(awk -F '\t' -v shape="$shape" \
    '$1 == shape && $3 == "grouped" { print $6 }' "$RESULTS")
  [[ -n "$off" && -n "$grouped" ]] || return 1
  case "$shape" in
    pp512) historical=330.0 ;;
    pp2048) historical=401.8 ;;
    pp4096) historical=417.6 ;;
    *) return 1 ;;
  esac
  delta_vs_off=$(awk -v grouped="$grouped" -v off="$off" \
    'BEGIN { printf "%+.1f%%", 100.0 * (grouped / off - 1.0) }')
  delta_vs_historical=$(awk -v grouped="$grouped" -v historical="$historical" \
    'BEGIN { printf "%+.1f%%", 100.0 * (grouped / historical - 1.0) }')
  echo "$shape grouped=$grouped tok/s off=$off tok/s delta-vs-off=$delta_vs_off historical-pipeline=$historical tok/s delta-vs-historical=$delta_vs_historical"
}

report_geometry() {
  local shape=$1
  local tokens chunk chunks
  tokens=$(awk -F '\t' -v shape="$shape" '$1 == shape { print $2; exit }' "$RESULTS")
  [[ -n "$tokens" ]] || return 1
  read -r chunk chunks < <(awk -v tokens="$tokens" '
    BEGIN {
      chunk = int((tokens + 7) / 8)
      if (chunk < 128) chunk = 128
      if (chunk > 4096) chunk = 4096
      start = 0
      while (start < tokens) {
        end = start + chunk
        if (end > tokens) end = tokens
        if (tokens - end > 0 && tokens - end < 16) end = tokens
        chunks++
        start = end
      }
      print chunk, chunks
    }
  ')
  echo "$shape tokens=$tokens effective-auto-chunk=$chunk chunks=$chunks"
}

main() {
  echo "=== Lever C perf $TS commit=$(git rev-parse HEAD)"
  echo "model=$MODEL"
  echo "protocol=N=$N interleaved, order alternated, one warmup per independent timed arm"
  local prompt
  for prompt in "$P512" "$P2048" "$P4096"; do
    echo "prompt=$prompt bytes=$(wc -c <"$prompt") sha256=$(sha256sum "$prompt" | awk '{print $1}')"
  done
  local rc=0
  (
    flock -w 21600 9 || {
      echo "LOCK TIMEOUT"
      exit 75
    }
    echo "lock acquired $(date -u +%FT%TZ)"
    snapshot
    local cell shape rep arm
    for cell in \
      "pp512:$P512" \
      "pp2048:$P2048" \
      "pp4096:$P4096"
    do
      shape=${cell%%:*}
      prompt=${cell#*:}
      for ((rep = 1; rep <= N; rep++)); do
        if (( rep % 2 == 1 )); then
          for arm in off grouped; do
            run_arm "$shape" "$prompt" "$arm" "$rep" || rc=1
          done
        else
          for arm in grouped off; do
            run_arm "$shape" "$prompt" "$arm" "$rep" || rc=1
          done
        fi
      done
    done
    snapshot
    echo "lock released $(date -u +%FT%TZ)"
    exit "$rc"
  ) 9>/tmp/memra-gpu.lock
  local perf_rc=$?

  printf 'shape\ttokens\tarm\tn\tmedian_s\tmedian_tok_s\n' >"$RESULTS"
  local shape arm
  for shape in pp512 pp2048 pp4096; do
    for arm in off grouped; do
      summarize_arm "$shape" "$arm" || perf_rc=1
    done
  done
  echo "=== medians ==="
  cat "$RESULTS"
  echo "=== effective naked PP-2 auto geometry ==="
  for shape in pp512 pp2048 pp4096; do
    report_geometry "$shape" || perf_rc=1
  done
  echo "=== comparisons ==="
  for shape in pp512 pp2048 pp4096; do
    report_comparison "$shape" || perf_rc=1
  done
  echo "=== Lever C perf rc=$perf_rc"
  echo "=== done $(date -u +%FT%TZ)"
  return "$perf_rc"
}

main >"$SUMMARY" 2>&1
rc=$?
echo "summary=$SUMMARY results=$RESULTS rc=$rc"
exit "$rc"
