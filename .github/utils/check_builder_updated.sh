lastref=`cat .github/utils/last_build_ref | tr -d '[:space:]'`
curref=`git ls-remote https://github.com/jayanthkoushik/build-webpage HEAD | cut -f1`
if [ "$lastref" = "$curref" ]; then
    echo '::set-ouput name=updated::0'
else
    echo '::set-ouput name=updated::1'
fi
