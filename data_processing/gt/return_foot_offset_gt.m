%% this function returns the foot angle offset (the bias from zero when the foot angle is in flat foot contact) for each subject

function foot_offset = return_foot_offset_gt(sub_string)

foot_offsets = [9.75535325541780
-2.87226878414971
1.14819136819837
1.12412002169708
-2.06198975493369
-4.94030798397037
-6.14679091668324
0.921437609251434
10.7164395177880
-6.94000579954376
-4.37193825771554
-1.37281670565447
-0.442765948111763
-6.03339586821992
-3.26371912008572
5.10795523010420
-4.94407625379629
-5.71305525705291
0.311926705221706
2.03909925474083
10.5603491341712
-10.8276357863813];

sub_ids = ["AB06"
"AB07"
"AB08"
"AB09"
"AB10"
"AB11"
"AB12"
"AB13"
"AB14"
"AB15"
"AB16"
"AB17"
"AB18"
"AB19"
"AB20"
"AB21"
"AB23"
"AB24"
"AB25"
"AB27"
"AB28"
"AB30"];

i = find(strcmp(sub_ids,sub_string));


foot_offset = foot_offsets(i);
% foot_offset = 0;

end
