# RecommenderSystemAlgo

Welcome to the competition reserved to the students of the Recommender Systems course in Politecnico di Milano.

Goal

The application domain is a music streaming service, where users listen to tracks (songs) and create playlists of favorite songs. The main goal of the competition is to discover which track a user will likely add to a playlist, based on

other tracks in the same playlist
other playlists created by the same user
Description

Please read carefully till the end of the page!

In this competition you are required to predict a list of 5 tracks for a set of playlists. The original unsplitted dataset includes around 1M interactions (tracks belonging to a playlist) for 57k playlists and 100k items (tracks). A subset of about 10k playlists and 32k items has been selected as test playlists and items. The goal is to recommend a list of 5 relevant items for each playlist. MAP@5 is used for evaluation. You can use any kind of recommender algorithm you wish (e.g., collaborative-filtering, content-based, hybrid, etc.) written in any language.

The prize

Each team will receive a final score according to the quality of recommendations computed on the private leaderboard, based on:

the position in the final leaderboard when the competition ends;
the position in the leaderboard every 2 weeks, during the competition;
the quality of recommendation in comparison to the baselines;
the size of the team.
The score is computed with the following formula:

score=baseline_bonus+standing_points+team_points
score=baseline_bonus+standing_points+team_points
Attention. Results on the public leaderboard are computed on a different subset of the test set, so it may differ from the private one.

Baseline bonus

You are provided with 8 baseline scores. Each baseline is computed with a different algorithm. If you are able to do better than n baselines, you will receive a bonus score that adds to your final score

b=2×n
b=2×n
Maximum baseline bonus is 16 points.

Important. The top most baseline (Algorithm LG DDTDM) does not count for scoring purposes.

Standing points

According to the standing in the private leaderboard, every two weeks points will be assigned to the teams, in the following manner:

si=18−18×log2[rank−1Nteams−1+1]
si=18−18×log2⁡[rank−1Nteams−1+1]
where i is the deadline,

Nteams=number of teams
Nteams=number of teams
and

rank=ranking of the team in the leaderboard=1..Nteams
rank=ranking of the team in the leaderboard=1..Nteams
Maximum standing vote is 18 points.

Important. You can register in any moment, regardless of the fact that one or more of the deadlines already passed. If you do not make any submission before the first deadlines, you will get 0 standing point for each of the missed deadlines.

Team points

Single-person teams receive one point of bonus

t=0 (two persons team)
t=0 (two persons team)
t=1 (one person team)
t=1 (one person team)
Final score

The total score is computed from the private leaderboard with the following formula:

score=∑iwi⋅si∑iwi+b+t
score=∑iwi⋅si∑iwi+b+t
where "i" is the i-th biweekly deadline, and

wi=1 (intermediate deadline)
wi=1 (intermediate deadline)
wi=2 (final deadline)
wi=2 (final deadline)
The last deadline weights twice each intermediate deadline.

Maximum final score is 35 points.

Attention. Results on the public leaderboard are computed on a different subset of the test set, so it may differ from the private one.

Enrolling to the competition

When registering on Kaggle, you must use your mail.polimi.it email address. If you use a different email address, your mark will not be registered.

Team merging

Team merging won't be allowed after November 14th. After the merging, for each of the past deadlines, the team will get the best final score of the single members.

For instance, suppose students A and B merge into the AB team. If student A got 24 and student B got 28 as final scores for the first deadline, the AB team will have 28 as score for the first deadline.

Attention. At the end of the competition, we will evaluate the activity and contributions of each team member. If we decide that a member has provided only a minimal contribution, we reserve the right to reduce or cancel his/her mark and, eventually, to add a bonus to the mark of the other member.

Team splitting

Team splitting is not allowed in any moment, unless you cancel your Kaggle account and create a new account with the same email address. In this case, you will lose all of your previous submissions (and the related points).

Deadlines

Deadlines will be every 15 days, on the following dates (at 23.59 CET):

15 October
1 November
15 November
1 December
15 December
15 January (final deadline)
