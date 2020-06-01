lastref=`cat .github/utils/last_build_ref | tr -d '[:space:]'`
curref=`git ls-remote https://github.com/jayanthkoushik/build-webpage HEAD | cut -f1`
reftest=`test $lastref != $curref`
echo '::set-ouput name=updated::${reftest}'
