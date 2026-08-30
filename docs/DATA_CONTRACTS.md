# Historical Data Contracts

## Security master

One immutable row per permanent security identifier:

~~~text
security_id, issuer_id, ticker, exchange, security_type, company_domain,
sector, listing_start, listing_end, delisting_return
~~~

Ticker is a dated label, not the primary key. Inactive and delisted securities remain in this table.

## Filing vintages

One normalized filing state per security and SEC accession:

~~~text
security_id, accession_number, accepted_at, period_end, ...
~~~

accepted_at is the availability clock. A historical decision may not use a row accepted after its cutoff. Amendments are new immutable vintages; they do not overwrite prior states.

## Raw prices

~~~text
date, security_id, close, volume, shares_outstanding
~~~

Prices are unadjusted. Dividends and splits must not already be embedded.

## Corporate actions

~~~text
date, security_id, action_type, value, qualified
~~~

Supported actions are DIVIDEND and SPLIT. Delisting dates and returns are controlled by the security master.

## Benchmarks

~~~text
date, benchmark, total_return
~~~

Every benchmark must cover the same observation dates as the tested portfolio.

## Factors

~~~text
date, MKT_RF, SMB, HML, RMW, CMA, RF
~~~

Returns must use decimal units and the same daily calendar as the portfolio.

## Fail-closed rules

The historical run fails when it detects future filings, duplicate filing accessions, duplicate prices, missing required delisting returns, invalid NAV, negative inventory, accounting-equation failures, or missing monthly vintage snapshots.

Tax payment source is explicit. Portfolio payment reduces strategy cash; external payment leaves portfolio NAV intact and reports after-tax total wealth net of the external taxes.
