# ADR 0001: Cornstarch/Oobleck ownership boundary

Cornstarch owns generic compile/activate/close primitives, model blueprints,
explicit stage specifications, and stable state manifests. Oobleck owns
profiling, heterogeneous composition, membership generations, failure policy,
replay, complete WORLD replacement, and state redistribution. The adapter is
lazy so generation plans and rank ownership remain usable before importing or
initializing distributed Cornstarch runtime state.
