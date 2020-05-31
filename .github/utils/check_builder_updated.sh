lastref=`cat .github/utils/last_build_ref | tr -d '[:space:]'`
curref=`git ls-remote git@github.com:jayanthkoushik/build-webpage HEAD | cut -f1`
test $lastref != $curref
