# Supplementary information source map

This joint source map is shared by both repositories. Single and cascade source paths refer to their respective repository, without creating a runtime dependency. This document maps the implementation actually retained in the integrated repositories. It is technical source material, not a claim that generic theory fills missing implementation details.

## S1 Overall workflow

DDP solves the complete realized horizon and also supplies ISO teacher actions. SDP fits seasonal classes/probabilities on TRAIN, solves a cyclic relative-value policy and evaluates on EVAL. ISO fits twelve monthly RFs, chooses settings on validation, refits on TRAIN and evaluates. Double DQN and cascade continuous PPO separate development/validation, final training and frozen evaluation. Single has four methods; PPO is cascade-only.

## S2.1 Physical model

Single:src/reservoir_benchmark/shared/reservoir_model.py,calculate_transition.
Cascade:src/cascade_reservoir_benchmark/environment.py,_transition,_transition_arrays,CascadeReservoirEnv.step.

S'=S+(I-Q)*seconds/1e9. N inflow is local interval natural inflow plus X total release, with zero monthly delay. H=max(0,(Zbegin+Zend)/2-tailwater). Turbine flow=min(total release,turbine capacity,installed kW/(8.5H)), with zero-head handling. Spill=total-turbine. PkW=8.5*turbine*H; EGWh=PkW*actual_month_hours/1e6.

Storage curves use linear interpolation; tailwater uses fourth-degree polynomial fitting and extrapolation. X adds high-flow anchors15666/1012.71 and20683/1016.64. X/N dead,normal,flood levels are1166/1240/1236 and765/812/804m. Turbine limits2340/3537m3/s, installed4200000/5850000kW, firm1854000/2406000kW; X total release max20683m3/s, N has no separate cap. End-target flood bounds apply May–October.30.4days is only a reward-scale reference.

Pseudocode:bound requested target; convert storage; enforce water availability/release handling; calculate actual end storage; tailwater/head/turbine/spill/power/energy; route upstream total release; repeat downstream; carry actual state. Extreme release/storage conflicts require explicit residual diagnostics, not an assumption that clipping always preserves mass.

## S2.2 DDP

Single:policy_generation/ddp/ddp_solver.py,optimize; ddp_acceleration.py caches transitions.
Cascade:ddp.py plus ddp_parallel.py and ddp_parallel_interpolated.py.

r=sum(E)-lambda*sum(P<Pfirm). Vt(s)=max_a[r+interpolated Vnext(T(s,a))], without RL discount. Single75levels, cascade75x48states,1m targets. Single exact target continuation; cascade bilinear interpolation in storage. Infeasible positive-weight interpolation corners remain-infinity. Hard final normal-level constraint at the end of the full4800month horizon, not annually.

Pseudocode:grids/limits; hard terminal value; reverse months; evaluate coupled transitions and continuation; ascending first-on-tie argmax; forward continuous reconstruction; diagnostics.

## S2.3 SDP

Single:policy_generation/sdp/sdp_probability.py,sdp_solver.py and evaluation/sdp/sdp_eval.py.
Cascade:sdp_probability.py,sdp.py,sdp_evaluation.py.

Ten equal-width classes per source/calendar month, TRAIN min/max, midpoint flows, empirical count/training-year probability. Cascade joint probability is the product of marginals, without contemporaneous or cross-month conditional transition estimation. Bellman maximization conditions on current classes; future class value is expected before next observation. Continue at actual continuous storage via linear/bilinear interpolation; discrete policy lookup uses nearest level indices.

Use mean training-calendar days during solution, actual days during evaluation. Maximum400iterations, residual diagnostic threshold1e-5GWh, stop on two stable policy passes. Saved single175 converged in5iterations; cascade400 in6. Requested targets may undergo physical repair. Do not describe the routine as DDP-style strict reachability filtering.

## S2.4 ISO

Single:policy_generation/iso_rf/policy.py,workflow.py.
Cascade:iso_models.py,iso_workflow.py.

Twelve RFs by month. Features:single[start level,current natural inflow]; cascade[two start levels,Xnatural inflow,Ninterval inflow]. Outputs:one/two same-month end targets. RF prediction is the tree-output average. No RF scaling or1m rounding. Use actual executed state in closed loop.

Both200trees,min_samples_leaf1,seed20260918. Single max_depth=None,max_features1.0; cascade16,0.5. Single anchor1200 development followed by validation50 lambda scan selecting700. Cascade fixed RF41 parameters and eleven teacher penalties select500; RF41 is a candidate label.

Single solves the development teacher on320years separately. Cascade solves a400year teacher before320/80row splitting and initializes validation from its1223/812m teacher state. Disclose this difference; neither substitute a newly computed teacher nor claim strict cascade teacher isolation.

## S2.5 Double DQN

Single:policy_generation/dqn/{dqn_agent,dqn_reservoir,dqn_runner}.py and evaluation/dqn/dqn_eval.py.
Cascade:dqn_{adapter,agent,training,evaluation}.py.

Masked online next-action argmax, target-network action value, SmoothL1 loss, Adam and uniform replay. Artificial240month truncation retains bootstrap with a real241st observation. Month is numeric minmax, not sine/cosine. Normalization uses learning data only and is frozen for evaluation.

Single3state input,75outputs,1536/768/384hidden withLayerNorm/ReLU,lr8e-4,gamma.97,buffer50000,batch128,warmup10000,epsilon.95to.01 over72000afterwarmup,target400learningupdates,gradient1,final2300episodes,lambda650,scale3064.32.

Cascade5state input,3600joint actions,action-conditioned Q=v(s)+dot(e_s,e_a)/sqrt(128)+b(a). State trunk1536/768/384; action features are two normalized target levels; action branches have128dimensional embeddings. lr1e-5,gamma.97,buffer50000,batch32,warmup10000,epsilon288000,target400,gradient1,final4800episodes,lambda5000,scale7332.48. Three seeds1,42,100. No reward clipping. Evaluation greedy, no learning.

## S2.6 Continuous PPO

Cascade continuous_ppo_{adapter,agent,training,evaluation}.py only. Old ppo_* is discrete historical implementation.

State5dimensions. Actor state trunk512/256 with128conditional heads; critic256/256. Latent Gaussian u; f=(tanh(u)+1)/2; target=L+f(U-L). Generate X first, preview release, derive N feasible interval and condition N action on X and geometry. Store the fraction-space squash log-prob correction exactly as implemented.

GAE delta=r+gamma*bootstrap*Vnext-V; recurrence stops across artificial windows while terminal bootstrap remains. Clipped policy ratio loss plus.5MSE value loss; no value clipping. lr9e-4 linearly to5%,gamma.97,GAE.85,clip.10,entropy0,maxgrad.5,batch320,max10epochs,targetKL.02 checked per epoch,Adam epsilon1e-5,logstd0 clamped[-5,1],squash epsilon1e-6.8envs x240steps,375updates,720000steps/seed,lambda6000. Evaluation deterministic latent means through the same transform.

## S3 Files and result definitions

configs/paper/reference_results.json maps each saved trajectory/JSON. workflows.json maps stages/configurations/artifact placement. provenance/source_manifest.json and release_manifest.json preserve origins and hashes. Cascade figure_registry.json maps current manuscript numbers to legacy entry IDs. New paths differ only in location; original JSON contents remain source evidence.

Energy=sum recorded monthly GWh/400. Reliability=monthly P>=firm. Minimum paper reliability=min(mean-seed station rates). Report simultaneous-station and summed-power reliability separately. Across-seed long-run energy SD usesddof1. Violin uses400calendar-year values after seed averaging. Monthly level quantiles pool seed-year observations. Spill converts monthly flow to volume using actual calendar seconds before aggregation. Cascade station-spill sum is not basin net water loss.

DQN/PPO target adjustment=abs(diff(r*_requested_target_level_m))within each seed,4799samples,includingDecember–January. ECDF lines average seed ECDFs; bands are seed min/max. Seasonal grouping uses the current month. The finalized scripts' statistics are retained.

DDP/SDP evaluation scans selected references on EVAL; their EVAL is not fully blind for representative selection. Current source evidence does not establish all historical training/selection chronology or a complete software lock; disclose those limits.
