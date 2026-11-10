#!/usr/bin/env bash
# Sample Gazebo's real-time factor beside running SITL flights: every 15 s, one sample per
# running flight container, appended to that flight's run folder as
# `rtf_samples.txt` ("<utc time> <rtf>"). Benchmark numbers are only comparable at real time,
# and the flight scripts do not record it themselves; `sim2real.route_follow.run_mission`
# reads the file into its tables (`RTF mean / min`, flights with a sample under 0.9 flagged).
#
#   sim2real/docker/rtf_sampler.sh &          # before a batch; kill it afterwards
#
# It finds the run folder from the container's /out mount and the world from PX4_GZ_WORLD,
# so it works for every runner (run_scenario, run_estimate, run_gazebo, run_mission). The
# first sample of a flight is near zero while Gazebo starts; readers skip it.
set -u
period="${RTF_PERIOD_S:-15}"
while true; do
    for c in $(docker ps -q); do
        out=$(docker inspect -f \
            '{{range .Mounts}}{{if eq .Destination "/out"}}{{.Source}}{{end}}{{end}}' "$c")
        world=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$c" \
            | sed -n 's/^PX4_GZ_WORLD=//p')
        [[ -z "$out" || -z "$world" ]] && continue
        v=$(timeout 10 docker exec "$c" gz topic -e -t "/world/${world}/stats" -n 1 2>/dev/null \
            | awk '/real_time_factor/{print $2; exit}')
        [[ -n "$v" ]] && echo "$(date -u +%FT%TZ) $v" >> "$out/rtf_samples.txt"
    done
    sleep "$period"
done
