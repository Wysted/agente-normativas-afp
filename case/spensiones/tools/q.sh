#!/bin/bash
# usage: [TENANT=spensiones] q.sh "<sql where-clause on unaccented text, e.g. t ~* 'a' and t ~* 'b'>" [limit] [width]  (text is unaccented as t)
# DB_CONTAINER overrides the compose db container (default agintis-spensiones-db-1).
TENANT=${TENANT:-spensiones}
W="$1"; L="${2:-10}"; N="${3:-300}"
docker exec -i "${DB_CONTAINER:-agintis-spensiones-db-1}" sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -At -v tenant="$0"' "$TENANT" <<SQL
select left(id::text,8)||' '||replace(profile,'spensiones_','')||' | '||left(regexp_replace(t,E'\\\\s+',' ','g'),$N) from (select id,profile,unaccent(text) t from chunks where tenant_id=:'tenant') x where $W limit $L;
SQL
