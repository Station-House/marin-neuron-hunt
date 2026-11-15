# L6-N9321: exemplars, positive side

Data: `pilot/fineweb-10m-sparse`. In each context the firing token is **[[bracketed]]**, newlines are shown as ⏎, and ⟨BOS⟩ marks a document start. Activation = silu(gate)·up, the input to `down_proj`.

This file is reading material for the draft PR, not an explanation. It will be removed before the PR is marked ready.

## Levels

| side | q99 | q999 | q9999 | extreme |
|---|---|---|---|---|
| + | 0.074 | 0.229 | 0.835 | +3.31 |
| − | 0.080 | 0.269 | 0.982 | -3.48 |

Fraction of tokens with a positive activation: 0.452.

**Logit lens** (what this neuron's output pushes the unembedding toward; at this early layer it is mostly noise):

- promotes: `' Sek'`, `' recreated'`, `'GW'`, `' Julian'`, `' Cho'`, `' Hog'`, `' Yaz'`, `' Sly'`, `' Thunder'`, `' yaz'`
- suppresses: `' minim'`, `' Movement'`, `'chat'`, `' =>\n'`, `' tide'`, `' Fridays'`, `' merc'`, `' Thurs'`, `' �'`, `'ц'`

## Positive side (+)

### Which tokens fire past q999 (3578 firings in 1477 documents)

This counts strong firings from the sparse tier, so it shows where the side fires, not where it stays quiet.

| token | firings | mean act |
|---|---|---|
| `' week'` | 879 | +0.57 |
| `' '` | 208 | +0.31 |
| `' weekend'` | 176 | +0.44 |
| `' Sunday'` | 176 | +0.57 |
| `' Saturday'` | 150 | +0.53 |
| `' Monday'` | 130 | +0.58 |
| `' weeks'` | 114 | +0.42 |
| `' Tuesday'` | 111 | +0.59 |
| `' Thursday'` | 107 | +0.56 |
| `' Friday'` | 101 | +0.57 |
| `' Wednesday'` | 99 | +0.54 |
| `'Monday'` | 77 | +0.55 |
| `'Tuesday'` | 69 | +0.49 |
| `'Sunday'` | 63 | +0.46 |
| `'Wednesday'` | 58 | +0.48 |
| `'Thursday'` | 56 | +0.55 |
| `' days'` | 46 | +0.31 |
| `'Friday'` | 41 | +0.45 |
| `' Mon'` | 40 | +0.54 |
| `'Saturday'` | 35 | +0.47 |
| `' weekly'` | 35 | +0.52 |
| `' night'` | 31 | +0.30 |
| `' weekends'` | 23 | +0.51 |
| `' Tue'` | 22 | +0.56 |
| `'-week'` | 21 | +0.39 |

### Top 32 exemplars

| rank | act | context |
|---|---|---|
| 0 | +3.31 | Jesus, come!⏎We need Your Light and Your Love now more than ever.⏎And earthy religions celebrate the Winter Solstice, the beginning of the ascendancy of the sun in the northern hemisphere on**[[ Monday]]**, December 21 at |
| 1 | +2.55 | the Santa Fe ski resort, Ski Santa Fe, seven times daily until New Year’s Day.⏎Normally, the district’s Blue Buses drive to the Santa Fe ski area three times a day during the**[[ week]]**, beginning from the South Capitol |
| 2 | +2.52 | in your behaviour and when things don't go to plan, adjust the plan; and I expect there will be high fives rather than haunts by the time December arrives again.⏎Christmas is one little**[[ week]]** away and I find myself in |
| 3 | +2.45 | a shirt, slop on some sunscreen and slap on a hat and then you would be⏎ready to go out into the sun. It is working because Marvin has worn his hat⏎every day this**[[ week]]** whilst he has been in the |
| 4 | +2.41 | Atlanta Symphony Orchestra with a glorious rendition of Handel’s “And the Glory of the Lord. Click here. Be sure to turn up your speakers and enter full screen.⏎Advent Day 17 ~**[[ Thursday]]** of the third week of Advent |
| 5 | +2.30 | be changed to the 27th of April, which is the king's birthday. In 2014 however it will be on the 26th of April because the 27th falls on a**[[ Sunday]]**. On this day the streets |
| 6 | +2.25 | <\|begin_of_text\|>I woke up one morning last**[[ week]]** and realized I’m not living |
| 7 | +2.16 | the children.⏎After this our father married Miss Platt, of Saybrook, by whom he had one daughter, who died at the age of eighteen.⏎I was born on the first day of February,**[[ Sunday]]** morning, 1801, |
| 8 | +2.12 | by Taylor R Ware This essay proves my love for dance. [317 words]⏎Imagination And Children by Lawson - [600 words]⏎Dead-To-The-World Man Walking by James Snyder One morning this**[[ week]]**, I woke up with a |
| 9 | +2.09 | <\|begin_of_text\|>Daily News -**[[ Thursday]]** 15 January 2015 |
| 10 | +2.02 | <\|begin_of_text\|>December 11, 2020 at 19:10 #28769⏎Burke Electrical Supply usually held its Christmas party on December fifteenth, partly because the**[[ week]]** before Christmas tended to be busy |
| 11 | +1.95 | D’s trolley-esque transit bus… but did you know that the downtown area has an ACTUAL trolley that you can climb aboard? See for yourself!⏎Mardi Gras this year was,**[[ Tuesday]]**, February 28, and |
| 12 | +1.91 | otes) will be our announced as the official Award Winner on August 22nd on every platform!⏎Come by for promotions and deals! We'd love to see you!⏎Store hours for BC Day:⏎**[[Sunday]]** July 31 10am |
| 13 | +1.87 | , the San Joaquin Valley—the swath of agricultural land that runs through central California—was designated a disaster area. Record-low runoff and scant rainfall had created drought conditions. At the beginning of Christmas**[[ week]]**, the weather was normal in |
| 14 | +1.80 | again somewhat suddenly and following a phone call, J was told that she could wind up at the end of the day with the assurance that she would be paid for the few weeks leading up to the annual**[[ week]]** she was scheduled to take at |
| 15 | +1.80 | many home foreclosures as new jobs.⏎Ubuntu on Thinkpad X60s⏎September 17, 2007⏎September 17, 2007⏎The site provides a nice daily (**[[week]]**days) review of the US |
| 16 | +1.79 | this trend continues in 2018, Ireland is likely to experience the highest number of new HIV diagnoses on record.⏎The global theme for World AIDS Day 2018 is ‘Know Your Status’. On**[[ Wednesday]]** 28th November in Le |
| 17 | +1.79 | : You know Candace, this is the second time I've climbed up there. I'm already down half a dress size.⏎- Candace's birthday is July 11, which occurs on a**[[ Friday]]** in this episode.⏎- Cand |
| 18 | +1.73 | <\|begin_of_text\|>4th Annual Depressing Song Competition**[[ Tuesday]]** 8/4 @ |
| 19 | +1.70 | to 10 a.m. at the Capital News Centre Indoor Track. For more information call John Moxness at 250-762-5446 or firstname.lastname@example.org.⏎Birthdays of the**[[ week]]** – Steve Nicholson (May |
| 20 | +1.69 | -tweeting that flashes across our computer screens, especially about politics. But are we gorging on empty calories?⏎To give you just a snapshot, let's look at one random morning this past**[[ Monday]]**.⏎Politico's "44 |
| 21 | +1.69 | ried fruit. If I do say so myself (I think hubby agrees) everything was delicious. Plus we had lots of leftovers – great, no cooking for the next few days!⏎The next morning,**[[ Friday]]**, we went out to breakfast |
| 22 | +1.67 | of the SMART4SEA Awards! SAFETY4SEA is thrilled to announce the 8 winners across 8 smart categories who walked with a SMART4SEA Award at the prestigious ceremony which took place on**[[ Tuesday]]** 30rd at the Y |
| 23 | +1.65 | the last year my website (4 years old) has been getting busier by the day thanks to Google. I felt that I needed to add new content 5 to 6 times a day (**[[week]]**end too!) to my website |
| 24 | +1.64 | . She became blind a few years after her husband’s death and was granted a handicapped parking space for her and her caretaker. To honor Delphine’s 112th birthday, the whole**[[ week]]** was announced as “Delph |
| 25 | +1.63 | , the easier it becomes. The easier it is, it is time to increase the challenge (in exercise, it means more repetitions / more weight). After 30 days (6 days of effort per**[[ week]]**), there is almost always a |
| 26 | +1.61 | ’s a very nice touch. Special thanks to Sriram, Sayyed and Manikandan for co-ordinating the event throughout....See More⏎Dinesh 6 months, 1**[[ week]]**, 2 days, |
| 27 | +1.60 | host an evening reception from 5 to 8 p.m.⏎Castleton Corners: Adults and children will find plenty of holiday activities at the St. Teresa School and Parish's annual Christmas fair on**[[ Sunday]]**, 9 a.m. |
| 28 | +1.60 | some speakers for the upcoming annual Internet Radio event of the year – the RAIN Summit. This event takes place each year during the NAB Show in Las Vegas, this year the one day conference is**[[ Monday]]** April 20th at the |
| 29 | +1.59 | : Third Annual Thanksgiving Cruise⏎Overall Member Rating⏎Third Annual Thanksgiving Cruise⏎This is our "Third Annual Thanksgiving Cruise" for my little family of five. We selected this cruise because it fell during the**[[ week]]** that we all are off of |
| 30 | +1.58 | <\|begin_of_text\|>Greetings from 53.5° where overnight on**[[ Saturday]]** the ambient air temperature and the |
| 31 | +1.56 | : OKC ZOO AND BOB MOORE SUBARU CELEBRATE INTERNATIONAL FLAMINGO DAY⏎Think pink and flock to the Oklahoma City Zoo for International Flamingo Day on**[[ Saturday]]**, April 24, from |

### Typical firings: 25 random ones between q999 and q9999 (|act| 0.23–0.84)

These show what the side does day to day, below the extremes.

| act | context |
|---|---|
| +0.54 | south towards Moe and then swinging back up towards a small town called Erica where we stopped for a late lunch and tripled the towns GDP for the**[[ week]]**.⏎Images by Alan |
| +0.23 | the immense natural beauty of the area along with a feast of live music from some of the very best concert artists in the world. Last year’s June**[[ concert]]** at Rayado was |
| +0.29 | 9 is set for a U.S. theatrical release of December 26th, 2008.⏎Wednesday, January 23, 2008⏎**[[Tuesday]]**, January 22 |
| +0.48 | 2012 as a WR and return man.⏎Day Three Target:⏎Kevin Norwood – Alabama (77)⏎This kid could be one of the top draft**[[ weekend]]** |
| +0.49 | , and Marci Levine will produce the movie.⏎9 is set for a U.S. theatrical release of December 26th, 2008.⏎**[[Wednesday]]**, January 23 |
| +0.26 | their institutions: some extending the call, others placing a new date on their call and even striking indefinitely.⏎Prisoners have been staying strong throughout these**[[ weeks]]** of the prison strike |
| +0.27 | away from Srinagar after incessant rains and cloudburst left many cattle dead and changed course of water of tributaries in the last**[[ week]]** of August.⏎The |
| +0.34 | 2010 tax deal … and helping to pass the American Taxpayer Relief Act of 2012.”⏎DECENT JOBS NUMBER (FOR NOW) –**[[ Friday]]**’s jobs report showing |
| +0.23 | clear, compassionate, honest, and non-violent communication skills⏎· Experience steadiness of mind through heart-centered listening⏎· Embrace freedom⏎The**[[ weekend]]** will include:⏎· |
| +0.28 | of the A. Jean Brugger Project.⏎Over the four-day clinic period, we saw 2,530 patients broken down as follows:⏎Monday**[[ Tuesday]]** Wednesday Thursday Total⏎ |
| +0.64 | s. But the confidence established by our sunrise vigil on the edge of the ICW carried us through the rest of the day.⏎One might think that**[[ weekend]]** boat traffic would interfere |
| +0.53 | this month’s Doodle, but it was only the 29th. I still had two days left! And no idea what to say anyhow.⏎**[[Wednesday]]** began with more sweating |
| +0.82 | corseted waistline and prominent bosom. In July of 1908, Les Modes reported on the fashions worn during La Grande Sem**[[aine]]** – the week that |
| +0.29 | classrooms, campus and culture with fellowship and fun. It invites costumes, and creativity while building community. As Student Council began to flesh out the daily Spirit**[[ Week]]** themes, they stumbled |
| +0.37 | to the public on all federal government work days from 8:30 a.m. to 4:30 p.m.; generally, it is open**[[ Monday]]** through Friday, excluding |
| +0.24 | Sebastian Vizcaíno left Mexico City on March 7, 1602, and arrived in Acapulco on the 19**[[th]]**. His expedition sailed |
| +0.72 | high resolution photos of 15th and 16th century armor are available to download from the Tinguely Museum website.⏎Submitted by Milica on**[[ Tue]]**, 2009 |
| +0.49 | 47:47 CEST 2013⏎Now it works! For download&oter info, the small infopage here .⏎Codecrypt project —**[[ Sun]]** Dec 2 |
| +0.79 | : the first problem was that I didn’t have $2000. And the second problem was that it was a little after 4 am on a**[[ Saturday]]** morning, and the |
| +0.68 | this year, and they want to have a shared meal at the last class, and reconvene on Monday January 31, 2022 (**[[Monday]]** February 7th |
| +0.23 | ina-style, through our great oozy pans of eggs and cheese served at an ungodly hour.⏎bless you each and everyone. ’til**[[ tomorrow]]**…⏎nina’s |
| +0.64 | , and the guys who didn’t have a problem with it were predictably sleazy.⏎Eighteen months of living a double life went by. The**[[ week]]** before Christmas, I |
| +0.54 | astronaut Mark Kelly is joining the company as its Director of Flight Crew Operations. The company announced that it will also begin taking reservations for tis flights this**[[ week]]**.⏎NBC News.com |
| +0.28 | I know but all things I love.⏎I got a new sales job which involved 2 weeks away from home training, and now 3 days a**[[ week]]** based at home, |
| +0.53 | <\|begin_of_text\|>This Month's Stories⏎December 5, 2007⏎December 2007**[[ Newsletter]]**⏎SBE Chapter |
