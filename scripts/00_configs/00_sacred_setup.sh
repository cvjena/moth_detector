
SACRED_CREDS=${SACRED_CREDS:-$(realpath ../../../sacred/config.sh)}

if [[ -f ${SACRED_CREDS} && -z ${NO_SACRED} ]]; then
	echo "Sacred credentials found (${SACRED_CREDS}); sacred enabled."
	source ${SACRED_CREDS}
else
	if [[ ! -f ${SACRED_CREDS} ]]; then
		echo "No sacred credentials found (${SACRED_CREDS}); disabling sacred."
	elif [[ ! -z ${NO_SACRED} ]]; then
		echo "NO_SACRED was set; disabling sacred."
	fi

	OPTS="${OPTS} --no_sacred"
fi
