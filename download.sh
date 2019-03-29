sudo apt-get install curl
mkdir qrpic
number=1000
signal=9999
while [ $number!=$signal ]
do
{
	curl http://qr.liantu.com/api.php?text=$number --output "./qrpic/"$number".png"
	number=`expr $number + 1`
}
done
