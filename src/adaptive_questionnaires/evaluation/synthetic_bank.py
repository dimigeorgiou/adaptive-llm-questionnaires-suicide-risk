"""Synthetic, non-clinical question bank for the V1-vs-V2 evaluation.

Every question carries a hidden ``intent`` label. Surface forms of one intent are
paraphrases, and some share few words on purpose, so ground-truth redundancy can be
measured without using the similarity function that V2 selects with. Each category
also has one "change event": a note text that signals a changed situation, plus
follow-up intents that clarify it.

This content was written for software evaluation. It is generic well-being
phrasing, not a validated instrument, contains no suicide-screening items, and must
not be used clinically.
"""

BANK = {
    "Routine": {
        "baseline_notes": ["Keeps a regular daily routine; sleep described as normal.",
                           "Daily routine regular; reports sleeping normally.",
                           "Routine stable, sleep normal according to the person."],
        "background": {
            "rt_days": ["How have your days been structured this week?", "What has a typical day looked like for you lately?",
                        "Can you walk me through an ordinary day this week?"],
            "rt_sleep": ["How has your sleep been over the past week?", "Have you been sleeping badly lately?",
                         "Has your sleep been poor recently?"],
            "rt_bedtime": ["What time have you usually been going to bed?", "When do you normally turn in at night these days?",
                           "Around what hour have you been going to sleep?"],
            "rt_meals": ["How have your meals fitted into your day?", "Have you been eating at regular times?",
                         "What has your eating pattern been like this week?"],
            "rt_morning": ["What has your morning routine looked like lately?", "How do your mornings usually start?",
                           "What do you do first after getting up?"],
            "rt_activity": ["How much have you been getting out of the house?", "Have you been going outside most days?",
                            "How often have you left home this week?"],
            "rt_evening": ["How do you usually spend your evenings?", "What do your evenings tend to involve lately?",
                           "What fills your time after dinner?"],
            "rt_predictable": ["What part of your day feels most predictable?", "Which part of the day is the most settled for you?",
                               "When in the day do things feel most stable?"],
        },
        "change": {
            "note": "Reports barely sleeping this week and skipping meals.",
            "followups": {
                "rt_ch_sleep": ["How has your sleep changed since last week?", "What has been different about your nights recently?"],
                "rt_ch_block": ["What has been getting in the way of sleeping?", "What keeps you awake at night at the moment?"],
                "rt_ch_meals": ["How have your meals been since your sleep changed?", "Which meals have you been skipping lately?"],
            },
        },
    },
    "Mood": {
        "baseline_notes": ["Mood described as steady.", "Reports a steady mood overall.", "Mood steady, no marked changes reported."],
        "background": {
            "md_general": ["How would you describe your mood over the past few days?", "How have you been feeling in general lately?",
                           "What word would sum up your mood this week?"],
            "md_lift": ["What has lifted your mood recently, even briefly?", "What has brought you a moment of enjoyment lately?",
                        "When did you last feel a bit better, and what helped?"],
            "md_low": ["When during the week did you feel most low?", "Which moments this week felt hardest emotionally?",
                       "At what point did you feel at your lowest?"],
            "md_energy": ["How has your energy been compared with last week?", "Have you felt more tired or more energetic recently?",
                          "How much energy have you had for daily tasks?"],
            "md_feelings": ["What feelings have been most present for you lately?", "Which emotions have come up most often this week?",
                            "What has been on your mind emotionally?"],
            "md_waking": ["How have you been feeling when you wake up?", "What is your mood like first thing in the morning?",
                          "How do you usually feel on waking?"],
            "md_worry": ["How much have you been worrying lately?", "Have worries been taking up a lot of your time?",
                         "What has been troubling you most?"],
            "md_calm": ["When did you last feel calm?", "What moments of calm have you had this week?",
                        "Where or when do you feel most at ease?"],
        },
        "change": {
            "note": "Describes feeling much lower than usual and withdrawn most days.",
            "followups": {
                "md_ch_since": ["How has your mood been since we last spoke?", "What has changed in how you feel since our last meeting?"],
                "md_ch_trigger": ["Did something happen that made you feel lower?", "Was there an event that affected how you have been feeling?"],
                "md_ch_withdraw": ["What has made you want to keep to yourself lately?", "How have you been spending time now that you feel withdrawn?"],
            },
        },
    },
    "Connection": {
        "baseline_notes": ["Weekly contact with a sibling.", "Sees a sibling once a week.", "Regular weekly contact with sibling reported."],
        "background": {
            "cn_spoken": ["Who have you spoken to this week?", "Which people have you been in contact with recently?",
                          "Who did you talk with over the past few days?"],
            "cn_overwhelmed": ["Who do you talk to when you feel overwhelmed?", "When things get too much, who do you turn to?",
                               "Who is there for you on difficult days?"],
            "cn_family": ["How have your contacts with family been lately?", "How are things with your relatives at the moment?",
                          "What has your family life been like recently?"],
            "cn_more": ["Is there someone you would like to be in touch with more?", "Who would you like to see more often?",
                        "Is there a relationship you would like to strengthen?"],
            "cn_connected": ["How connected have you felt to the people around you?", "Do you feel close to others at the moment?",
                             "How much have you felt part of a group or community?"],
            "cn_reach": ["What has made it easier or harder to reach out to others?", "What stops you from contacting people?",
                         "What helps you get in touch with others?"],
            "cn_support": ["Who has supported you recently?", "Where have you found support this week?",
                           "Who helped you when you needed it lately?"],
            "cn_lonely": ["How often have you felt lonely recently?", "Have you felt alone much this week?",
                          "When do you tend to feel most isolated?"],
        },
        "change": {
            "note": "Has stopped seeing the sibling after an argument.",
            "followups": {
                "cn_ch_argument": ["How are things with your sibling since the argument?", "What happened between you and your sibling?"],
                "cn_ch_other": ["Who else have you been in touch with since you stopped seeing your sibling?",
                                "Has anyone else been around for you since the argument?"],
                "cn_ch_repair": ["What would it take to reconnect with your sibling?", "Would you like things with your sibling to change?"],
            },
        },
    },
    "Coping": {
        "baseline_notes": ["Uses walking to manage stress.", "Manages stress by going for walks.", "Walking reported as main way of coping."],
        "background": {
            "cp_helped": ["What has helped you get through difficult moments this week?", "What got you through the hard times recently?",
                          "When things were tough, what made a difference?"],
            "cp_tried": ["Which coping strategies have you tried recently?", "What have you done to manage stress lately?",
                         "Which ways of handling pressure have you used?"],
            "cp_builds": ["What do you usually do when stress builds up?", "How do you react when tension rises?",
                          "What is your first response when you feel under pressure?"],
            "cp_stressful": ["What has been the most stressful part of your week?", "Which situation has put you under the most strain?",
                             "What caused you the greatest pressure recently?"],
            "cp_easier": ["What would make next week a little easier?", "What could help the coming days go more smoothly?",
                          "What small change might ease things next week?"],
            "cp_less": ["Which of your usual strategies has worked less well lately?", "Has anything that normally helps stopped working?",
                        "What used to help but does not seem to now?"],
            "cp_rest": ["How have you been resting and recovering?", "What gives you a break from everyday pressures?",
                        "When do you manage to relax?"],
            "cp_ask": ["How comfortable are you asking for help when stressed?", "Do you find it easy to ask others for help?",
                       "What makes asking for support difficult?"],
        },
        "change": {
            "note": "Could not go walking this week because of an injury; stress higher.",
            "followups": {
                "cp_ch_alt": ["When walking was not possible, what else helped?", "What have you done instead of walking this week?"],
                "cp_ch_hard": ["What has been hardest to cope with since the injury?", "How has the injury affected the way you handle stress?"],
                "cp_ch_plan": ["What could replace walking while you recover?", "Which other activities might help until you can walk again?"],
            },
        },
    },
    "Goals": {
        "baseline_notes": ["Wants to return to a part-time course.", "Plans to go back to a part-time course.", "Goal: resume part-time studies."],
        "background": {
            "gl_forward": ["What is one thing you are looking forward to?", "What are you hoping to enjoy in the coming days?",
                           "Is there something ahead that you are excited about?"],
            "gl_small": ["What small goal would you like to work on this week?", "Which modest aim could you set for the next few days?",
                         "What little step would you like to take soon?"],
            "gl_matters": ["What matters most to you at the moment?", "What is most important in your life right now?",
                           "Which things feel most meaningful to you currently?"],
            "gl_progress": ["What progress have you noticed, however small?", "What has gone a bit better recently?",
                            "Where have you seen improvement lately?"],
            "gl_month": ["What would you like to be different a month from now?", "How would you like things to look in a few weeks?",
                         "What change would you most like to see soon?"],
            "gl_purpose": ["What activity has given you a sense of purpose recently?", "What has felt worthwhile for you lately?",
                           "When did you last feel useful or needed?"],
            "gl_course": ["How are your plans for the course going?", "What steps have you taken toward your studies?",
                          "How close are you to starting the course?"],
            "gl_obstacle": ["What stands in the way of your goals right now?", "What obstacles are you facing with your plans?",
                            "What makes it hard to move forward at the moment?"],
        },
        "change": {
            "note": "Enrolment for the course was refused; feels discouraged about plans.",
            "followups": {
                "gl_ch_refused": ["How did you feel when the enrolment was refused?", "What went through your mind when you heard about the course?"],
                "gl_ch_next": ["What options are open now that the course is not possible?", "What could be a next step after the enrolment news?"],
                "gl_ch_keep": ["What would help you keep going with your plans?", "What could restore some hope about your goals?"],
            },
        },
    },
}

CATEGORIES = list(BANK)
