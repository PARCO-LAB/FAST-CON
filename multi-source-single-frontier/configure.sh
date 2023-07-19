
threads=`./build/getResidentThreads 0`

if grep -q -P "MAX_CONCURR_TH\t\t$threads" config.h; then
	exit
else
	echo "\nConfiguration with "$threads" threads\n"

	sed -E -i "s/(MAX_CONCURR_TH\t\t)[^=]*$/\1$threads/" config.h
fi
