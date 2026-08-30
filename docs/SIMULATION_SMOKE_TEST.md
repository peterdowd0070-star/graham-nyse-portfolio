# Simulation Smoke Test

## Permitted use

The generated fixture verifies software behavior only. It is not a backtest and must not be used to estimate return, risk, alpha, drawdown, or tax efficiency.

## Tested behavior

- historical membership is determined from listing start and end dates;
- an inactive security remains eligible before its delisting and disappears afterward;
- a held delisted security is liquidated using its recorded delisting return;
- filing snapshots exclude acceptances after the decision cutoff;
- new filings appear in later monthly snapshots;
- dividends and splits are explicit events;
- raw prices are used without adjusted-price double counting;
- March and September rebalance incumbents;
- June and December reconstruct the portfolio;
- all four scenarios run through all six weighting strategies;
- position and sector caps remain satisfied;
- taxable modes maintain tax lots and wash-sale basis adjustments;
- unsupported narrative numbers fail report validation.

## Censoring policy

Generated NAV, holdings, trades, scores, and performance statistics remain under gitignored output paths. No synthetic performance number is retained in repository documentation. Test assertions concern invariants and event behavior, not whether generated returns are positive.

## Command

~~~bash
./scripts/run_local_validation.sh
~~~
