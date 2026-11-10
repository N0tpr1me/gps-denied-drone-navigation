#!/usr/bin/env bash
# Build the Basalt image inside the Claude cloud container, which reaches the internet only
# through an HTTPS proxy with its own CA (sim2real/docker/cloud_build.sh does the same for
# the SITL image). Everywhere else (the GPU server, the Mac) use the plain build:
#   docker build -t dronenav-basalt eval/vio_basalt
#
#   sudo dockerd >/tmp/dockerd.log 2>&1 &     # once per container
#   eval/vio_basalt/cloud_build.sh            # ~25 min on 4 cores
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${BASALT_IMAGE:-dronenav-basalt}"
CA="${CLOUD_CA_BUNDLE:-/root/.ccr/ca-bundle.crt}"
[[ -f "${CA}" ]] || { echo "proxy CA not found: ${CA}" >&2; exit 1; }

# The real Dockerfile with a proxy-trust layer right after FROM, so apt, pip and git in the
# build trust the proxy's certificate.
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
awk '
    { print }
    /^FROM / && !done {
        print "COPY --from=cloudca ca-bundle.crt /usr/local/share/ca-certificates/cloud-proxy.crt"
        print "RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \\"
        print "    && update-ca-certificates && rm -rf /var/lib/apt/lists/*"
        print "ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \\"
        print "    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \\"
        print "    GIT_SSL_CAINFO=/etc/ssl/certs/ca-certificates.crt"
        done = 1
    }' "${HERE}/Dockerfile" > "${tmp}/Dockerfile"
cp "${CA}" "${tmp}/ca-bundle.crt"

docker build --network host -t "${IMAGE}" -f "${tmp}/Dockerfile" \
    --build-context "cloudca=${tmp}" \
    --build-arg "HTTP_PROXY=${HTTP_PROXY:-${http_proxy:-}}" \
    --build-arg "HTTPS_PROXY=${HTTPS_PROXY:-${https_proxy:-}}" \
    --build-arg "http_proxy=${HTTP_PROXY:-${http_proxy:-}}" \
    --build-arg "https_proxy=${HTTPS_PROXY:-${https_proxy:-}}" \
    --build-arg "NO_PROXY=${NO_PROXY:-localhost,127.0.0.1}" \
    "$@" "${HERE}"
