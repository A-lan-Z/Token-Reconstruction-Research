# TRR-P02 task-owned publication inventory

This is the canonical task-owned publication set for the completed exploratory
public diagnostic. It includes the frozen plan, final report and state,
review receipts, the exact-case capture, the retained run4 outputs, and all
preserved failed-attempt receipts. Raw per-row values remain in the retained
diagnostics artifact; this inventory does not duplicate them. Root owns the
publication commit and pull request.

| path | status | bytes | SHA-256 |
| --- | --- | ---: | --- |
| `coordination/requests/TRR-P02.md` | preserved incoming packet; byte-identical to source | 9156 | `e16db54941ba85dd7f9f9e930578e9412ee18d348b1dc4c8746a9a25e103cad5` |
| `experiments/TRR-P02/setup/pull-request-body.md` | root-owned final PR body | 2229 | `aa7988b2aa4e7d580ffb2570793c4b2b38afcfe5aaedde66cfab1f3a855a81b9` |
| `coordination/results/TRR-P02.md` | final four-decision result | 8482 | `7e65f3181efffc8bc6233ef3b572f1d970afd873d603a33f68c5da983293c917` |
| `coordination/parallel/TRR-P02.json` | compact completion state/index | 10160 | `f1617501bf975c7b792b2be7f1c02f07956d6fc456780f2434f5dbd289dccb95` |
| `experiments/TRR-P02/manifest.json` | aggregate structured evidence manifest | 27624 | `b3bbced66d9b6cd314471d2c641fe3d54b502f493ab58d005e695bc586a6f1e7` |
| `experiments/TRR-P02/plan.json` | frozen execution plan used by run4 and case capture | 6923 | `9f191d0376d29a9bd46241060f5466738d55b375beb1baf78cb920b72594e030` |
| `experiments/TRR-P02/setup/resource-preflight.md` | host/resource/public-asset preflight | 5559 | `c1b38e47ca693e607d179a4147a7b7dec3bbf5c0e4b258436c7a848503be08de` |
| `experiments/TRR-P02/setup/public-diagnostic-exclusion.template.json` | empty helper template | 1833 | `0e9741f499da03ec1451bf5e11f5f2dab6cc1119a7c1f761daa30782530c62e4` |
| `experiments/TRR-P02/setup/public-diagnostic-exclusion.final.json` | immutable run4 exact-case exclusion capture | 42997 | `3b671dea06371834dfaf8863fd2b667fb2894f82d171d29d314236ec7abaa6dc` |
| `experiments/TRR-P02/review/design.md` | reviewed diagnostic design | 12664 | `0bf075e992065d8f07d74ceae208f1f0ab3e5fe65196aa0565867f57fb6068f1` |
| `experiments/TRR-P02/review/validation.md` | model-free validation receipt | 4019 | `13cffa78dd72f04a53e79f24872b83842dd0098bcf3cd8f33c279c52f3afdaf6` |
| `experiments/TRR-P02/review/retry-validation.md` | retry and failure validation receipt | 7929 | `f50c6799e1ff039465c10ab4f04a7f820ba9a92ced4fc112d2fa364425e36ef0` |
| `experiments/TRR-P02/review/run4-summary.md` | retained implementation numerical summary | 4855 | `4274204c8bdd87f312724b871679065ff43ffb5300d02572331fe620b6c907aa` |
| `experiments/TRR-P02/review/final-results-audit.md` | independent final-results audit | 12619 | `4ff16c2fb10a4e3d4e3916ce0d6c923afbc995f214601b2b3ec6959800fdc0f3` |
| `experiments/TRR-P02/review/derive_summary_figure.py` | reviewed summary-figure derivation script | 4167 | `b7d26bfb45120811860c8bb177151a2013c26f6efaaec0df0a280669015dbe33` |
| `experiments/TRR-P02/review/summary_geometry.png` | reviewed summary geometry/ranking figure | 137359 | `0f4160a97590ef0ab56813ef8dca954d587e872fe6e27b4896701b043341f7e1` |
| `src/token_reconstruction/trr_p02/exclusion.py` | model-free exact-case capture helper | 9677 | `c5dbf59c521818cbcac566f0b0232e7ecf0adebaca8f5363e2d066c0932ff08d` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run4/preflight.json` | retained run4 preflight | 2283 | `62edea9f94579b527ed89ac8f9c0aee5bc54555aadcc3df2c251a56bc5147431` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run4/qualification.json` | C6 all-eight endpoint qualification | 884 | `d9351b137fb62f0d6d9f7c7b02b10d8758e92f26eb775d97ae0924abef5500ca` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run4/activation_panel.safetensors` | 46-row public activation tensor | 2046040 | `e63026f56063083fe009fe3211548875310dd3295e7c205f0e3759f1ae5a15ca` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run4/offset_geometry.png` | retained offset figure | 44740 | `6796c56602983dac2340e6ae8992e7adc2246ff621908bbb0c3b46d07129e721` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run4/lens_geometry.png` | retained lens figure | 77999 | `fc03d6f48d19c694ac7bc5bbac716b08ad85e9bc95881a61a2265cb217698cc7` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run4/diagnostics.json` | retained numerical diagnostics | 635370 | `7352573df457804b2702a419571a9feb100ae5863d32238eab6f38f19a9586c4` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run4/run_manifest.json` | retained runner manifest | 5816 | `2ad5a6049c988940acbe0e1ef4b62320ad094c1b2e1673ca6f8e5edcc7f7f710` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run4/outer-receipt.txt` | retained outer receipt | 74 | `91c1d5653039db559f8a710a245cae307af9041722a7ed14d745aef6106ef045` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run4/outer-stderr.txt` | retained outer stderr | 147 | `0831a5ae240e73da475a9b4470763a2a137998e5467e742e753edf00a2158f48` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run4/outer-stdout.txt` | retained outer stdout | 394 | `97691608a98e69832682d11e82ba4d6d5e8b6e2a961b99a1f2cb30302f0ad2b3` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run4/outer-time-v.txt` | retained `/usr/bin/time -v` receipt | 1502 | `a450d15bc32093eae8933a004509e71064115559add11d2af43f07842155bed1` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run4-launch-failure.json` | excluded pre-Python wrapper failure | 676 | `6295147889eb6a8aa980d59af7255f9cf2eca7b8f9a81ce9f79490447370c273` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run1/failure-receipt.json` | excluded run1 lens-import failure | 3530 | `de13ed7fa7b24973d7141bb3130a773f07c0e3def6269b9af9ba879e884af43b` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run1/preflight.json` | run1 preflight receipt | 2283 | `34426cec309081e1b38945059e2f7a1d57246808bdbfbe6bd276edcc7f5c38a9` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run1/outer-receipt.txt` | run1 outer receipt | 74 | `aa8211d69dcf16188475153c441ea87080ab7696ec070ec638540a06198f0c5b` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run1/outer-stderr.txt` | run1 outer stderr | 1163 | `630956438f80f251b1fb3f815461b04787cb9927d07dd1930de0397ef7047a04` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run1/outer-stdout.txt` | run1 outer stdout | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run1/outer-time-v.txt` | run1 `/usr/bin/time -v` receipt | 1536 | `115783ba7506525047bc7bd6c737241df2ddd1088d4b79fa6eda2d48f3214baa` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run2/failure-receipt.json` | excluded run2 figure-generation failure | 4140 | `59fc9ba3b1805ef6961580ddecf9a89895b0eedb3fac6e253924fe453a514600` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run2/preflight.json` | run2 preflight receipt | 2283 | `46199ced846a057cc767b6b12dea548ffbe8ecd07ca1540c87d65cc5ae40a7c0` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run2/qualification.json` | run2 C6 qualification receipt | 884 | `b0b7e70236b8a3c0090ee8212ae8790a47cef6a7174d34db07a0a93b558d4fbc` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run2/outer-receipt.txt` | run2 outer receipt | 74 | `96d5bafe457ea203335598deebc88bb1be05f1aa9af8d0f3e45ba0372b0479f7` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run2/outer-stderr.txt` | run2 outer stderr | 1514 | `ab34b967440ce26e05b9201aa1ba77f5025364d6a6d1fa8640be4f73bf5b2745` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run2/outer-stdout.txt` | run2 outer stdout | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run2/outer-time-v.txt` | run2 `/usr/bin/time -v` receipt | 1538 | `d752c3ff266bd2248beff120a46e87bb553910931542167e81be29308e08895c` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run3/failure-receipt.json` | excluded run3 serialization failure | 4521 | `308052ed547a4266671d6172f4c5a4fdbaef4c475465b6b6432743dc630106aa` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run3/preflight.json` | run3 preflight receipt | 2283 | `1f318a493f403ca88c3adf33b0fbf6c582737adbefd674f4f33410ee717357b8` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run3/qualification.json` | run3 C6 qualification receipt | 884 | `eabf455bbb5ea1fe0f2b3bebd110231f07377cf24892a1da0e621bf2de60c9d3` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run3/offset_geometry.png` | run3 failed-attempt figure | 44740 | `6796c56602983dac2340e6ae8992e7adc2246ff621908bbb0c3b46d07129e721` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run3/lens_geometry.png` | run3 failed-attempt figure | 77999 | `fc03d6f48d19c694ac7bc5bbac716b08ad85e9bc95881a61a2265cb217698cc7` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run3/outer-receipt.txt` | run3 outer receipt | 74 | `f24f310c645e50209d2ca5bfdd68ffd560529de8f7f5fca3495915a156e7f61c` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run3/outer-stderr.txt` | run3 outer stderr | 1132 | `e0634e46dae01e4221aa503e1838d20df3a54cb9cc89a192d1e0484a120127fc` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run3/outer-stdout.txt` | run3 outer stdout | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run3/outer-time-v.txt` | run3 `/usr/bin/time -v` receipt | 1539 | `d7b86aada52e0b46f4a86725492944d8a2152063af25d84d4ecf22cdfe94e4c6` |
| `scripts/trr_p02/diagnose_geometry.py` | reviewed diagnostic source or focused test | 73740 | `ff7e11223efc3316eda7aa206519ec68791259d43ac312f932345977160f9dfc` |
| `src/token_reconstruction/trr_p02/geometry.py` | reviewed diagnostic source or focused test | 20085 | `80967a3c738efdca64d716763b52ecc00a8795ab61be0527926b7b8734bdeaf2` |
| `src/token_reconstruction/trr_p02/__init__.py` | reviewed diagnostic source or focused test | 697 | `585eb453394b59b9f0d432ca35acda1a47aeeddf1eb01c625179ce816a9637be` |
| `tests/test_trr_p02_geometry.py` | reviewed diagnostic source or focused test | 6263 | `ff808e793f6b0bc9e44ea6294436eadac0292ae4b2f195e2b74c230500bba72d` |
| `experiments/TRR-P02/setup/publication.json` | sanitized PR publication and verification receipt | 1182 | `78a1f0c55429f87ae8ca845e03f762cb762fa0d07f279c5b2774635a2a4826fb` |

The inventory intentionally omits its own digest because a file cannot contain
a stable self-hash. Record the inventory SHA-256 externally in the publication
commit or pull-request handoff. Duplicate staging copies of the run4 outer
receipts and local interim drafts are excluded from this canonical list.
