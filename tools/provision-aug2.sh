#!/usr/bin/env bash
# provision-aug2.sh — hour-zero automation for the Aug 2-3 8xH100 capacity block.
# Wraps tools/provision-8x.sh (the on-box flow from the Jul 31 box) with the launch
# phase + the idle-safety termination reminders this window requires.
#
# BLOCK: cr-02251913b8f9ea0a6  p5.48xlarge  us-east-2a (use2-az1)
#        2026-08-02T11:30Z -> 2026-08-03T11:30Z  (HARD END — AWS terminates the box)
# Verified 2026-08-01 (read-only describes): reservation state=scheduled (payment
# cleared), AMI available, SG/subnet/key all live in us-east-2.
#
# Subcommands:
#   preflight              read-only sanity (safe anytime): reservation state, AMI,
#                          SG, subnet, key pair
#   launch                 run-instances into the block (refuses unless state=active,
#                          i.e. at/after 2026-08-02T11:30Z), wait for the public IP,
#                          open the Mumbai SG for box-to-box rsync, push the provision
#                          scripts, and kick off `onbox` under nohup on the box
#   onbox <mumbai-ip>      RUN ON THE BOX: install the termination motd + cron walls
#                          (T-2h/T-30/T-15/T-5), receipts convention (~/receipts),
#                          then the standard on-box flow (provision-8x.sh), then the
#                          nvcc plumbing the fresh DLAMI lacks
#   sync-repo <box-ip> [src-dir]
#                          rsync a LOCAL checkout onto the box as ~/memra (plain tree,
#                          not a git repo — same convention as every prior box) and
#                          stamp ~/memra/BOX-COMMIT.txt with the source commit
#
# All ids/regions are BAKED LITERALS (workflow args do not propagate — bake per-run
# params as script literals). Mission map: tools/box-aug2-mission.md
set -euo pipefail

# ---- this window's literals -------------------------------------------------------
REGION=us-east-2
CR_ID=cr-02251913b8f9ea0a6
BLOCK_END_UTC="2026-08-03T11:30Z"                 # hard end; walls fire 09:30/11:00/11:15/11:25Z
AMI=ami-08e0eea8869a74b3e                          # DLAMI "OSS Nvidia Driver" PyTorch 2.9 Ubuntu 24.04
                                                   # 20260722 — ships NO cuda toolkit (/usr/local/cuda
                                                   # absent); provision-8x.sh copies ~/cuda-13.3.1 from
                                                   # Mumbai. 28T NVMe pre-mounted at /opt/dlami/nvme.
ITYPE=p5.48xlarge
KEY=darklanes-bench                                # key-06eb7939282418e02, exists in us-east-2
SG=sg-0da8bd046bfadec47                            # darklanes-8x-ssh (tcp/22 0.0.0.0/0)
SUBNET=subnet-0e80dd54d46959a91                    # default us-east-2a == use2-az1 (must match block AZ)
NAME=darklanes-8x-aug2
ROOT_GB=300                                        # AMI default root is 20 GiB — far too small for
                                                   # cuda toolchain + cargo target trees; gp3 300 GiB
                                                   # costs ~\$1.3 for the 24h window
PEM=~/.ssh/darklanes-bench.pem
MUMBAI_IP=13.203.16.133                            # darklanes-bench (i-0439a6725788bec0c) — toolkit,
                                                   # repo and models rsync source; KEPT as board-reference
MUMBAI_SG=sg-06696c61b777c64a2                     # darklanes-bench-sg, ap-south-1
MUMBAI_REGION=ap-south-1
STALE_BOX_CIDR=18.117.231.105/32                   # dead Jul-31 box (i-0fdece58750076e2f) — still in the
                                                   # Mumbai SG as of Aug 1; launch swaps it for the new IP

SSH_OPTS=(-i "$PEM" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no)

cr_state() {
  aws ec2 describe-capacity-reservations --region $REGION \
    --capacity-reservation-ids $CR_ID \
    --query 'CapacityReservations[0].State' --output text
}

cmd_preflight() {
  echo "== capacity block $CR_ID =="
  aws ec2 describe-capacity-reservations --region $REGION --capacity-reservation-ids $CR_ID \
    --query 'CapacityReservations[0].{State:State,Type:InstanceType,AZ:AvailabilityZone,Start:StartDate,End:EndDate}' \
    --output table
  echo "== ami / sg / subnet / key =="
  aws ec2 describe-images --region $REGION --image-ids $AMI \
    --query 'Images[0].{Name:Name,State:State}' --output table
  aws ec2 describe-security-groups --region $REGION --group-ids $SG \
    --query 'SecurityGroups[0].GroupName' --output text
  aws ec2 describe-subnets --region $REGION --subnet-ids $SUBNET \
    --query 'Subnets[0].AvailabilityZone' --output text
  aws ec2 describe-key-pairs --region $REGION --key-names $KEY \
    --query 'KeyPairs[0].KeyName' --output text
  echo "== mumbai rsync source =="
  aws ec2 describe-instances --region $MUMBAI_REGION \
    --filters Name=ip-address,Values=$MUMBAI_IP \
    --query 'Reservations[0].Instances[0].State.Name' --output text
  echo "preflight OK — launch is legal once state=active (block start 2026-08-02T11:30Z)"
}

cmd_launch() {
  local state; state=$(cr_state)
  if [ "$state" != "active" ]; then
    echo "REFUSING to launch: block state=$state (instances launch only while the block is"
    echo "active, i.e. from 2026-08-02T11:30Z). Re-run then. Nothing was created."
    exit 1
  fi

  echo "== run-instances into $CR_ID =="
  local iid
  iid=$(aws ec2 run-instances --region $REGION \
    --image-id $AMI --instance-type $ITYPE --key-name $KEY \
    --security-group-ids $SG --subnet-id $SUBNET \
    --instance-market-options MarketType=capacity-block \
    --capacity-reservation-specification "CapacityReservationTarget={CapacityReservationId=$CR_ID}" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$ROOT_GB,\"VolumeType\":\"gp3\"}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
    --count 1 --query 'Instances[0].InstanceId' --output text)
  echo "instance: $iid — waiting for running"
  aws ec2 wait instance-running --region $REGION --instance-ids "$iid"
  local ip
  ip=$(aws ec2 describe-instances --region $REGION --instance-ids "$iid" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  echo "public ip: $ip"

  echo "== mumbai SG: swap stale box rule for $ip/32 (box-to-box rsync) =="
  aws ec2 revoke-security-group-ingress --region $MUMBAI_REGION --group-id $MUMBAI_SG \
    --protocol tcp --port 22 --cidr $STALE_BOX_CIDR 2>/dev/null \
    && echo "revoked stale $STALE_BOX_CIDR" || echo "stale rule already gone"
  aws ec2 authorize-security-group-ingress --region $MUMBAI_REGION --group-id $MUMBAI_SG \
    --protocol tcp --port 22 --cidr "$ip/32" 2>/dev/null \
    && echo "authorized $ip/32" || echo "rule for $ip/32 already present"

  echo "== waiting for sshd =="
  local up=""
  for _ in $(seq 1 40); do
    if ssh "${SSH_OPTS[@]}" -o ConnectTimeout=5 ubuntu@"$ip" true 2>/dev/null; then up=1; break; fi
    sleep 10
  done
  [ -n "$up" ] || { echo "ssh never came up — box booted but unreachable; investigate"; exit 1; }

  echo "== push provision scripts + start onbox (nohup, detached — inline ssh nohup is flaky"
  echo "   without </dev/null; lesson from the board-run gotcha) =="
  scp "${SSH_OPTS[@]}" "$(dirname "$0")/provision-8x.sh" "$(dirname "$0")/provision-aug2.sh" ubuntu@"$ip":~
  ssh "${SSH_OPTS[@]}" ubuntu@"$ip" \
    "nohup bash ~/provision-aug2.sh onbox $MUMBAI_IP </dev/null > ~/provision-aug2.log 2>&1 & echo onbox started"

  cat <<EOF

LAUNCHED: $iid @ $ip  (block hard-ends $BLOCK_END_UTC — reminders install on-box)
  watch provisioning : ssh ${SSH_OPTS[*]} ubuntu@$ip tail -f provision-aug2.log
  pin the repo state : bash tools/provision-aug2.sh sync-repo $ip   # from the local rig,
                       AFTER provisioning finishes (it overwrites the Mumbai-rsync'd ~/memra)
  mission map        : tools/box-aug2-mission.md
EOF
}

cmd_onbox() {
  local mumbai=${1:?usage: provision-aug2.sh onbox <mumbai-ip>}
  [ -d /opt/dlami/nvme ] || echo "WARNING: /opt/dlami/nvme missing — not the expected DLAMI?"

  echo "== idle-safety FIRST: termination reminders (block hard-ends $BLOCK_END_UTC) =="
  # dynamic motd: hard-end + live countdown at every login
  sudo tee /etc/update-motd.d/99-capacity-block >/dev/null <<'MOTD'
#!/bin/sh
end=$(date -ud "2026-08-03 11:30:00" +%s); now=$(date -u +%s)
mins=$(( (end - now) / 60 ))
echo "*** CAPACITY BLOCK cr-02251913b8f9ea0a6 HARD-ENDS 2026-08-03T11:30Z (~${mins} min left) ***"
echo "*** AWS terminates this box at end-time. Receipts -> ~/receipts, rsync back by T-30. ***"
echo "*** Nothing in /tmp survives. GPU 0 = BENCH ONLY. ***"
MOTD
  sudo chmod +x /etc/update-motd.d/99-capacity-block
  # wall broadcasts: T-2h (phase-3 start), T-30, T-15, T-5 (all UTC; box tz is UTC)
  sudo tee /etc/cron.d/capacity-block-end >/dev/null <<'CRON'
CRON_TZ=UTC
30 9  3 8 * root wall "CAPACITY BLOCK ENDS 11:30Z — 2 HOURS LEFT. Phase 3 now: receipts scp'd + committed, board cells, box-sweep (tools/box-aug2-mission.md)."
0  11 3 8 * root wall "CAPACITY BLOCK ENDS 11:30Z — 30 MINUTES. rsync ~/receipts back NOW. Nothing in /tmp survives termination."
15 11 3 8 * root wall "CAPACITY BLOCK ENDS 11:30Z — 15 MINUTES."
25 11 3 8 * root wall "CAPACITY BLOCK ENDS 11:30Z — 5 MINUTES. Final box-sweep: /tmp, ~ stray dirs, rsync-only trees."
CRON
  sudo chmod 644 /etc/cron.d/capacity-block-end

  echo "== receipts convention: everything measurable lands under ~/receipts (never /tmp) =="
  mkdir -p ~/receipts

  echo "== standard on-box flow (provision-8x.sh: rust + cuda-13.3.1 + repo + models from $mumbai) =="
  if [ -f ~/provision-8x.sh ]; then bash ~/provision-8x.sh "$mumbai"
  else bash ~/memra/tools/provision-8x.sh "$mumbai"; fi

  echo "== nvcc plumbing (fresh DLAMI has no /usr/local/cuda) =="
  # validate-h100.sh defaults MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc; m0 scripts use
  # /usr/local/cuda/bin/nvcc — cover both without lying about versions:
  grep -q MEMRA_NVCC ~/.bashrc || echo 'export MEMRA_NVCC=$HOME/cuda-13.3.1/bin/nvcc' >> ~/.bashrc
  [ -e /usr/local/cuda ] || sudo ln -sfn "$HOME/cuda-13.3.1" /usr/local/cuda

  echo "ONBOX DONE — mission map: ~/memra/tools/box-aug2-mission.md"
  echo "GPU 0 = BENCH ONLY. Lanes export CUDA_VISIBLE_DEVICES explicitly."
}

cmd_sync_repo() {
  local ip=${1:?usage: provision-aug2.sh sync-repo <box-ip> [src-dir]}
  local src=${2:-$(cd "$(dirname "$0")/.." && pwd)}
  local commit; commit=$(git -C "$src" log -1 --format='%H %s' 2>/dev/null || echo "not-a-git-checkout")
  echo "== rsync $src -> ubuntu@$ip:~/memra (plain tree, not a git repo) =="
  rsync -az -e "ssh ${SSH_OPTS[*]}" --exclude 'target*' --exclude '.git*' "$src/" ubuntu@"$ip":~/memra/
  ssh "${SSH_OPTS[@]}" ubuntu@"$ip" "echo '$commit' > ~/memra/BOX-COMMIT.txt && cat ~/memra/BOX-COMMIT.txt"
  echo "synced; rebuild on-box before any battery (stale-binary lesson: VERIFY-GATE false-FAILs)"
}

case "${1:-}" in
  preflight) cmd_preflight ;;
  launch)    cmd_launch ;;
  onbox)     shift; cmd_onbox "$@" ;;
  sync-repo) shift; cmd_sync_repo "$@" ;;
  *) sed -n '2,30p' "$0"; exit 1 ;;
esac
