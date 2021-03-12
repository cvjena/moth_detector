
if [[ -f sacred_creds.sh ]]; then
	echo "Sacred credentials found; sacred enabled."
	source sacred_creds.sh
else
	echo "No sacred credentials found; disabling sacred."
	OPTS="${OPTS} --no_sacred"
fi
