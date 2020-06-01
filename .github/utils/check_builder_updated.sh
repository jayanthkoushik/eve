lastref=`cat .github/utils/last_build_ref | tr -d '[:space:]'`
curref=`git ls-remote https://github.com/jayanthkoushik/build-webpage HEAD | cut -f1`
if [ "$lastref" = "$curref" ]; then
    updated=0
else
    updated=1
fi
echo '::set-ouput name=updated::${updated}'
