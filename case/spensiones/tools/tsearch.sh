#!/bin/bash
# usage: [TENANT=spensiones] tsearch.sh '<regex>' [snippet_limit]  -- accent- and case-insensitive search over all chunks of the tenant
# DB_CONTAINER overrides the compose db container (default agintis-spensiones-db-1).
TENANT=${TENANT:-spensiones}
RX="$1"; N="${2:-0}"
docker exec -i "${DB_CONTAINER:-agintis-spensiones-db-1}" sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -At -v rx="$0" -v tenant="$1"' "$RX" "$TENANT" <<SQL
select profile||' '||count(*) from chunks where tenant_id=:'tenant' and unaccent(text) ~* unaccent(:'rx') group by profile order by 1;
$( [ "$N" -gt 0 ] && echo "select left(id::text,8)||' '||replace(profile,'spensiones_','')||' | '||regexp_replace(substring(unaccent(text) from '(?i).{0,110}'||unaccent(:'rx')||'.{0,160}'),E'\\\\s+',' ','g') from chunks where tenant_id=:'tenant' and unaccent(text) ~* unaccent(:'rx') limit $N;" )
SQL
