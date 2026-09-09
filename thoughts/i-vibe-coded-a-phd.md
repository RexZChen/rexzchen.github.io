i already did a phd once, which apparently was not enough, so i made a browser game where you can do it again.

[US CS PhD Simulator](https://rexzchen.github.io/PhD-Simulator/) is a satirical little academic operating system: apply to definitely-fictional universities, pick an advisor archetype, manage research and writing, survive deadlines, watch your stipend meet rent, and try to remain a mammal until graduation.

it is also vibe coded.

## why would i do this

because phd life has the structure of a management game already.

you have limited energy. every project takes longer than the optimistic estimate. your advisor may be caring, chaotic, famous-but-absent, or running a small academic empire. papers move from idea to experiment to draft to submission to reviews, except sometimes reviewer 2 sends them directly to character development.

meanwhile, coursework, teaching, internships, networking, life admin, money, stress, confidence, and the human body all insist they are not side quests.

i wanted to turn that whole strange system into something you can click through. not a realistic predictor of anyone's career, so please do not cite it in an admissions decision, but recognizable enough that people who have been there occasionally need to close the tab and breathe.

## the vibe-coding part

the fun thing about building with an LLM is that the distance between “this would be a funny mechanic” and “there is now a button doing it badly” becomes very short.

the dangerous thing is exactly the same.

the model can scaffold a UI, propose event text, connect state transitions, and refactor a surprising amount of code. but it does not own the taste of the game. it does not know which joke is affectionate, which system is tedious, which failure feels unfair, or whether a number is balanced after forty turns. those decisions stay stubbornly human.

so the loop was roughly:

- describe a mechanic
- get a playable version unusually fast
- discover five interactions neither of us anticipated
- fix the state logic
- rewrite the joke
- play it again
- somehow add achievements

this is basically software development, only with a very enthusiastic pair programmer who has never paid rent on a stipend.

## what is in there

you can start from different backgrounds, browse a suspiciously familiar set of US computer-science programs, and choose among advisor personalities with their own tradeoffs. then time advances while you decide whether to research, write, teach, network, chase an internship, handle life admin, or, controversially, rest.

there are projects, venue deadlines, submissions, rebuttals, rejections, acceptances, prelims, proposals, the dissertation, the job market, random events, and a growing collection of achievements. the interface is bilingual in English and Chinese. everything runs in the browser.

some of my favorite lines are tiny bits of academic compression. “estimates are a genre.” “you are allowed to be a mammal.” a code link returning 404 “is itself a result.”

none of this is autobiographical, legally speaking.

## what i learned

vibe coding is excellent at buying iteration speed. that matters for a game because you do not discover the design by staring harder at a specification; you discover it by playing the bad version and noticing what should feel different.

but faster iteration does not remove engineering. once the prototype becomes a real stateful system, consistency matters: a delayed event must still make sense months later, a saved game must survive schema changes, and one new modifier must not quietly destroy twelve probability calculations.

the code can arrive quickly. coherence still has to be earned.

anyway, [go ruin your academic work-life balance in a controlled environment](https://rexzchen.github.io/PhD-Simulator/).

publish, perish, or take a nap.
