import unittest
import math
import numpy as np
import time

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from markevaluate import MarkEvaluate

class TestMarkEvaluate(unittest.TestCase):

    BERT_HIDDEN_SIZE = 768

    sentences0 = ['This framework provides an easy method to compute dense vector representations for sentences, paragraphs, and images.',
                  'This framework generates embeddings for each input sentence',
                  'sentences0 are passed as a list of string.',
                  'The quick brown fox jumps over the lazy dog.',
                  'This naming convention informs the test runner about which methods represent tests',
                  'The remainder of the documentation explores the full feature set from first principles.',
                  'With this framework, one is able to test these metrics in an exhausitve way.'] * 3

    sentences1 = ["The move is expected to delay the country's vaccination programme by several weeks.",
                  "Drug watchdog the European Medicines Agency last week announced a possible link with clots but said the risk of dying of Covid-19 was much greater.",
                  "Several European countries had previously briefly suspended the jab.",
                  "Most have now resumed vaccinations with AstraZeneca, but often with limits to older age groups.",
                  "South Africa has also paused its use, despite the Johnson & Johnson being its preferred vaccine because of its effectiveness against the South African variant",
                  "Both vaccines work by a similar method, known as adenoviral vectors.",
                  "Danish officials said that all 2.4 million doses of the AstraZeneca vaccine would be withdrawn until further notice."] * 3

    sentences01 = ['This framework provides an easy method to compute dense vector representations for sentences, paragraphs, and images.',
                   'This framework generates embeddings for each input sentence']

    sentences11o = ["This action will delay the country's vaccination programme for some time",
                    "Drug watchdog the European Medicines Agency announced last week a possible link with clots but said the risk of dying of Covid-19 was much greater."]

    sentences11o = ["This.",
                    "Drug."]

    sentences11 = ["The move is expected to delay the country's vaccination programme by several weeks.",
                   "Drug watchdog the European Medicines Agency last week announced a possible link with clots but said the risk of dying of Covid-19 was much greater."]
    # https://www.bbc.com/news/world-europe-56744474

    def test_markevaluate1(self):

        me = MarkEvaluate.MarkEvaluate()
        self.assertEqual(me.estimate(cand=self.sentences0, ref=self.sentences0)['Petersen'], 1, msg="Test Petersen Estimator using Theorem A.1.")

    def test_markevaluate2(self):

        me = MarkEvaluate.MarkEvaluate()
        res = me.estimate(cand=self.sentences0, ref=self.sentences0)
        self.assertEqual(res['Schnabel_qul'], 1, msg="Test Schnabel (quality) Estimator using Theorem A.2.")
        self.assertEqual(res['Schnabel_div'], 1, msg="Test Schnabel (diversity) Estimator using Theorem A.2.")

    def test_markevaluate2_2(self):

        me = MarkEvaluate.MarkEvaluate(orig=True)
        res = me.estimate(cand=self.sentences0, ref=self.sentences0)['Schnabel_qul']
        self.assertTrue(0 <= res and res <= 1, msg="Test Schnabel (quality) Estimator (original).")
        res = me.estimate(cand=self.sentences0, ref=self.sentences0)['Schnabel_div']
        self.assertTrue(0 <= res and res <= 1, msg="Test Schnabel (diversity) Estimator (original).")

    def test_markevaluate3(self):

        me = MarkEvaluate.MarkEvaluate()
        self.assertEqual(me.estimate(cand=self.sentences0, ref=self.sentences0,)['Schnabel_qul'], 1, msg="Test Schnabel Estimator using Theorem A.2.")

    def test_markevaluate3_3(self):

        me = MarkEvaluate.MarkEvaluate(orig=True)
        res = me.estimate(cand=self.sentences0, ref=self.sentences0)['Schnabel_qul']
        self.assertTrue(0 <= res and res <= 1, msg="Test Schnabel (quality) Estimator (original).")
        res = me.estimate(cand=self.sentences0, ref=self.sentences0)['Schnabel_div']
        self.assertTrue(0 <= res and res <= 1, msg="Test Schnabel (diversity) Estimator (original).")

    def test_markevaluate4(self):

        me = MarkEvaluate.MarkEvaluate()
        self.assertEqual(me.estimate(cand=self.sentences0, ref=self.sentences0)['CAPTURE'], 1, msg="Test CAPTURE Estimator using Theorem A.3.")

    def test_markevaluate4_4(self):

        me = MarkEvaluate.MarkEvaluate(orig=True)
        res = me.estimate(cand=self.sentences0, ref=self.sentences0)['CAPTURE']
        self.assertTrue(0 <= res and res <= 1, msg="Test CAPTURE Estimator (original).")

    def test_markevaluate5(self):

        me = MarkEvaluate.MarkEvaluate()
        result = me.estimate(cand=self.sentences0, ref=self.sentences1)
        self.assertTrue(0 <= result['Schnabel_qul'] and result['Schnabel_qul'] <= 1, msg="Test different input with different topics and different lengths.")
        self.assertTrue(0 <= result['Schnabel_div'] and result['Schnabel_div'] <= 1, msg="Test different input with different topics and different lengths.")

    def test_markevaluate6(self):

        me = MarkEvaluate.MarkEvaluate(sent_transf=False)

        ex_sample = [
            "Hello World.",
            "This is a test for the ME-Metrik."
        ]

        len_token = 0

        for ex in ex_sample:
            len_token += len(me.tokenizer.tokenize(ex))

        len_token_set = len_token * 5
        embds = me.get_embds(sentences=ex_sample)
        x, y = embds.shape
        assert x == len_token_set
        assert y == self.BERT_HIDDEN_SIZE

    def test_markevaluate7(self):

        me = MarkEvaluate.MarkEvaluate(sent_transf=False)

        ex_sample = self.sentences0

        len_token = 0

        for ex in ex_sample:
            len_token += len(me.tokenizer.tokenize(ex))

        len_token_set = len_token * 5
        embds = me.get_embds(sentences=ex_sample)
        x, y = embds.shape
        assert x == len_token_set
        assert y == self.BERT_HIDDEN_SIZE

    # def test_markevaluate8(self):

    #     me = MarkEvaluate.MarkEvaluate(sent_transf=False)
    #     me.estimate(cand=self.sentences01, ref=self.sentences11)

    def test_markevaluate9(self):

        me = MarkEvaluate.MarkEvaluate(sent_transf=False, sntnc_lvl=True)
        re = me.estimate(cand=self.sentences01, ref=self.sentences11)

    def test_markevaluate10(self):
        me = MarkEvaluate.MarkEvaluate(sent_transf=False, sntnc_lvl=True)
        re = me.estimate(cand=self.sentences11o, ref=self.sentences11)

    def test_markevaluate11(self):
        me = MarkEvaluate.MarkEvaluate(sent_transf=False, sntnc_lvl=True, orig=True)
        re = me.estimate(cand=self.sentences11o, ref=self.sentences11)

    def test_markevaluate12(self):
        me = MarkEvaluate.MarkEvaluate(sent_transf=False, sntnc_lvl=True, orig=True)
        re = me.estimate(cand=self.sentences0, ref=self.sentences1)

    def test_markevaluate13(self):
        s0 = [
            "This framework provides an easy method to compute dense vector representations for sentences, paragraphs, and images."
        ]
        s1 = [
            "The move is expected to delay the country's vaccination programme by several weeks."
        ]
        me = MarkEvaluate.MarkEvaluate(sent_transf=False, sntnc_lvl=True, orig=True)
        re = me.estimate(cand=s0, ref=s1)

    # def test_markevaluate14(self):
    #     s0 = [
    #         "This framework provides an easy method to compute dense vector representations for sentences, paragraphs, and images."
    #     ]
    #     s1 = [
    #         "The move is expected to delay the country's vaccination programme by several weeks."
    #     ]
    #     me = MarkEvaluate.MarkEvaluate(
    #         sent_transf=False, sntnc_lvl=True, orig=True)
    #     re = me.estimate(cand=self.sent_r_0, ref=self.sent_r_1)

    def test_markevaluate15(self):
        s0 = [
            "This framework provides an easy method to compute dense vector representations for sentences, paragraphs, and images."
        ]
        s1 = [
            "The move is expected to delay the country's vaccination programme by several weeks."
        ]
        me = MarkEvaluate.MarkEvaluate(
            sent_transf=True)
        re = me.estimate(cand=s0, ref=s1)
        assert me.data_org.k == 0
        

    sent_r_0 = ['In broiling heat on the western tip of Croatia, Patrick Vieira is demanding that little bit more from his young Manchester City players.',
        '‘Come on, come on, come on,’ he urges, with increasing cadence, as his elite development squad — or reserve team, in old money — play a two-touch, six-a-side game.',
        '‘It’s hot, you are tired.',
        'Keep the ball.',
        'Never lose it.’ Vieira’s proteges, along with the Under 18 group, coached by former Blackburn winger Jason Wilcox, are in the quaint coastal town of Novigrad for a 10-day training camp set against the panoramic backdrop of the Adriatic Sea.',
        'VIDEO Scroll down to watch City prospect Devante Cole in action for England U19s .',
        'English core: Patrick Vieira lends some advice to young Manchester City star Devante Cole .',
        'Father figure: The development squad have been training in Zudigrad ahead of the new season .',
        'Discussion: Vieira talks with Under 18s manager Jason Wilcox at a City youth fixture .',
        '‘It’s not an army barracks but it’s also not five-star luxury and glam,’ says Mark Allen, head of the academy.',
        '‘It offers the boys a taste, but we keep them grounded.’ ‘We have a motto here,’ Wilcox begins.',
        '‘Great person, great footballer.',
        'That means punctuality, appearance, work ethic, respect.',
        'You speak to cleaners how you speak to the manager.',
        '‘You shake hands with every member of staff in the morning and when they leave at night.',
        'It’s vital.’ At the club’s Carrington training ground in Manchester, discipline is instilled in these young men.',
        'If they forget an item of kit, whether it be their water bottle or shin pads, they will not train.',
        'If they are late for a team meeting on match-day, they will not play.',
        'Team spirit: Man City are looking to bring youngsters through to accord with Financial Fair Play .',
        'Teamwork: The former midfielder chats to Christian Lattanzio during a training session .',
        'Angus Gunn .',
        'Age: 18, Goalkeeper .',
        'He has endured some injury problems but re-emerged as one of the most exciting talents.',
        'Ashley Smith-Brown .',
        'Age: 18, Defender .',
        'Assured as a full back, centre-half or central midfielder, he has already trained under Vieira and played for England Under 16s and Under 18s.',
        'Tosin Adarabioyo .',
        'Age: 16, Defender .',
        'A commanding centreback, powerful and composed in possession.',
        'As a 15-year-old last season was regular in the the Under 18s defence.',
        'Brandon Barker .',
        'Age: 17, Left winger .',
        'Direct and fast with a terrific left foot.',
        'In Jason Wilcox has the perfect mentor as a left winger.',
        'Devante Cole .',
        'Age: 19, Forward .',
        'The son of Andrew, he can operate from the wing or up front.',
        'Was excellent in the UEF A Youth League last season.',
        '‘It is basic good manners,’ Allen says, his face gleaming with pride.',
        '‘Socks are rolled up, shirts are tucked in.',
        'I have a great picture where there are two players about to come on for England at youth level.',
        'You can tell immediately which one is a City player.',
        'His shirt is tidy, his shorts are right, shinpads are correct.',
        'It shows they are listening.’ There is a conscious effort to shield these teenagers from the trappings of fame.',
        'While adidas are already sponsoring some of City’s English 17-year-olds and agents have free rein to handpick the region’s finest talent, they are also given cookery and driving awareness classes.',
        'City are looking to nurture players from the cradle to the gravy train of the Premier League but in the boardroom they recognise that improvement is required.',
        'Since the Abu Dhabi takeover in the summer of 2008, no player has graduated from the City academy to cement a place in the first team squad.',
        'Last September, City defeated Manchester United with 10 overseas, outfield players.',
        'English players Jack Rodwell — who signed for Sunderland on Tuesday — and Scott Sinclair started nine Premier League matches between them since signing two years ago.',
        '‘We want to bring talent through our academy into our first team,’ says Vieira.',
        '‘There are no borders in football but if there are seven or eight Manchester boys, then fantastic.’ Potential: Angus Gunn and Ashley Smith-Brown could be stars of the future at Manchester City .',
        'Youngsters: City also have high hopes for Brandon Barker and\xa0Tosin Adarabioyo .',
        'Upbringing: The EDS team encourage their men to be great people as well as great footballers .',
        'Certainly, the potential is there: 30 of the 46 players in Croatia hail from the UK or Ireland.',
        'Light blue is increasingly the dominant colour in many of England’s young dressing rooms.',
        '‘We had seven under-16s in the England squad last year,’ Wilcox reveals.',
        '‘That was a record for us.',
        'We have some incredibly talented English players.',
        'Brandon Barker, Ashley Smith-Brown, Angus Gunn, Kean Bryan, Tosin Adarabioyo.',
        'The players are coming through.',
        '‘Below the under-18 group, over 90 per cent of our academy is English.',
        'Recently we offered eight professional contracts and six of those are English boys.',
        '‘Four of those six are local boys.',
        "It’s the ideal scenario but the wider you spread the net, the more chance you have of finding the gem.’ In charge: Wilcox on the touchline in a Man City Under 18 fixture at Everton's Goodison Park .",
        'Way back when: Wilcox, in the colours of Blackburn Rovers, takes on Vieira (right), of Arsenal, in a Premier League match during the 1998-1999 season .',
        'The seeds of talent are beginning to germinate.',
        'Last season, City’s Under 11s and Under 14s were national champions and the Under 18 side were northern league winners.',
        'Under Vieira, an Under 19 side reached the quarter-final of the UEFA Youth League.',
        'A 6-0 victory over Bayern Munich reverberated around Europe, with five English names on the team sheet.',
        'Most are yet to be seen in the first team squad, something Vieira attributes to the ‘massive’ gap between youth competitions and the demands of elite football.',
        'It is why City are thought to remain receptive to discussions over B-teams in the lower tiers.',
        'As the authorities prevaricate, City are single-minded in their aspiration and little encapsulates the journey from chip-fat to caviar quite like the money and dedication flowing into this academy.',
        'Time and effort: Vieira insists he can shape stars of the future .',
        'Having invested heavily in the first team, securing two Premier League titles in three years, Sheik Mansour is now hard at work on City’s foundations.',
        'Later this year, the £150m City Football Academy will open, a stone’s throw from the Etihad Stadium, where Tony Blair once intended to build a Super Casino.',
        'It will boast facilities unrivalled in the English game and Rick Owen, a club kit man for more than 20 years, reflects: ‘We used to train on council pitches and do pre-season at a school between Stoke and Crewe.',
        'How times change.’ On this summer morning, it is a breathless training session in sticky, cloying conditions and Vieira has become irritated, noticing that his players have become attracted to the ball.',
        '‘Stop, stop,’ he orders, his players freezing instantly.',
        '‘Look at yourselves.',
        'Ten of you, all bunched together!',
        'How can you play like this?',
        'Look for the space, make the pitch bigger.’ He motions, spreading out his hands.',
        '‘The boys need to understand this,’ Vieira insists.',
        '‘When you have the ball, the pitch must be as big as possible.',
        'If you lose it, make it as tight as possible and then seven seconds, maximum, to win it back.',
        "Foreign imports: Manuel Pellegrini's first team has been almost impossible to break into in recent years .",
        'Star: Sergio Aguero is one of many big-money moves City have made .',
        '‘The best teams have a quick recovery.',
        'When you press, it is the whole team, high and fast, even the goalkeeper.',
        'Watch Manuel Neuer — unbelievable, he is like an old No 5.',
        'But he wasn’t born this way, he trained hard.',
        'If we start early, we can create these players.’ This, in a nutshell, is the club’s philosophy: an intoxicating brand of high-tempo, passing football that has been outlined by Allen, sporting director Txiki Begiristain and academy director Brian Marwood.',
        'It is the identity that City now encourage at all levels, from the Under-11 group to the first team under Manuel Pellegrini.',
        'Allen expands: ‘When I took the job four years ago I outlined a 10-year plan to really start to see a group of talent coming through together all playing the City way.',
        '‘Cycles take time.',
        'In the late 90s it was France, then Spain, now Germany.',
        'Our moment will arrive.’ ‘The numbers will not lie,’ Vieira concedes, puffing out his cheeks.',
        '‘We have to make a report in 10 years on how many players are in the first team.',
        'Then we can say we did it right or we did it wrong.’ VIDEO Liverpool v Manchester City highlights .']

    sent_r_1 = ['from broiling heat on City western his of Croatia, Patrick Vieira Manchester demanding more young bit that In tip little is the players.',
        '‘ Come on, come on, come on,’ he increasing, old urges cadence, a touch elite development reserve— or squad team, in with money— six as two- his, play- a- side game.',
        '‘ hot ’s It, are you tired.',
        'Keep ball the.',
        'Never lose with.’ in Novigrad proteges, along it panoramic Under 18 group, set Wilcox former Blackburn winger Jason by, against Vieira the 10-day coastal for of ’s town a quaint training camp coached are the the backdrop of the Adriatic Sea.',
        'in Scroll down City for to prospect Devante Cole VIDEO action watch EnglandU19s.',
        'some young: City Vieira lends English Cole to core Manchester Patrick star Devante advice.',
        'Father figure: been ahead squad new The training in season development of the have Zudigrad.',
        'Discussion: fixture talks with Jason 18s manager Under at Wilcox City a youth Vieira.',
        '‘ academy the not an army barracks also it ’s but not head- star says and glam,’ luxury Mark Allen, five of ’s It.',
        '‘ It Wilcox boys the here taste, keep we but them grounded.’‘ We have a motto a,’ offers begins.',
        '‘ person Great, great footballer.',
        'means That punctuality, appearance, ethic work, respect.',
        'to the You cleaners how you speak to speak manager.',
        '‘ in night hands with every member of and You morning the staff when they leave at shake.',
        'It ground vital.’ At in club ’s Carrington training ’s the instilled, discipline young Manchester in these is men.',
        'whether they bottle an or of kit, If it shin their will forget item be pads, they water not train.',
        'If day a will play are team meeting on match- they, they late not for.',
        'Team City: to spirit are through Man bring youngsters looking to with accord Financial Fair Play.',
        'Teamwork: Christian former midfielder chats to The session a during training Lattanzio.',
        'Angus Gunn .',
        'Goalkeeper: 18, Age.',
        'endured has He some most as but re - the problems one of emerged injury exciting talents.',
        'Brown Smith- Ashley.',
        'Defender: 18, Age.',
        'Assured as or he has, trained- half a for midfielder, full back already centre under Vieira and played central 16s Under England and Under 18s.',
        'Tosin Adarabioyo .',
        'Age: Defender, 16.',
        'in commanding centreback, possession and composed A powerful.',
        'in a season- old last 15-year defence regular As the the Under 18s was.',
        'Brandon Barker .',
        'Age: Left, 17 winger.',
        'with and fast Direct foot terrific left a.',
        'as Jason left winger the perfect mentor In a Wilcox has.',
        'Devante Cole .',
        '19: Age, Forward.',
        'or son up Andrew, he can operate wing the from The of front.',
        'last A in League UEF excellent Youth the Was season.',
        '‘ basic is It says face,’ Allen good, his manners gleaming with pride.',
        '‘ Socks are shirts in, rolled are tucked up.',
        'youth have are there for where great a two players about come to on picture England at I level.',
        'one can tell player a You is which City immediately.',
        'His correct shorts tidy, his is are right, shinpads are shirt.',
        'shows It they teenagers shield.’ There these a conscious fame to listening is are from the trappings of effort.',
        'English talent finest already free driving of cookery ’s While awareness- olds and agents have sponsoring they to handpick the region ’s are adidas, rein are also given City and some 17-year classes.',
        'required are nurture to looking recognise from gravy cradle to the the train of the in that but Premier the boardroom they players League improvement is City.',
        'team no squad of summer in the takeover Dhabi 2008, the player has graduated from the City academy cement to a place in the first Since Abu.',
        'Last September, defeated City 10 with United Manchester overseas, outfield players.',
        'Rodwell for signing English— who signed players them Scott Tuesday— and on Sinclair started nine Premier two matches between Sunderland since Jack League years ago.',
        '‘ first want into bring academy through our talent to our We team,’ says Vieira.',
        '‘ There are in borders no football City future there are seven or the Potential boys, then fantastic.’ Manchester: Angus Gunn of at Smith- Brown be could stars and eight if Ashley Manchester but.',
        'hopes: also City \xa0 high Youngsters for Brandon Barker and have Tosin Adarabioyo.',
        'encourage: as EDS team Upbringing men their to be great people The footballers as great well.',
        'there, in potential is Certainly: 30 of the 46 players the or Ireland from the UK Croatia hail.',
        'Light colour is increasingly the dominant blue ’s England rooms many in young dressing of.',
        '‘ We had Wilcox the in under-16s England squad year last,’ seven reveals.',
        '‘ for was record a That us.',
        'players have talented incredibly some English We.',
        'Bryan Tosin, Ashley Smith- Brown, Gunn Angus, Kean Brandon, Barker Adarabioyo.',
        'The through are coming players.',
        '‘ group the under-18 Below, our 90 English cent of over academy is per.',
        'boys professional offered eight we contracts and six of English are those Recently.',
        '‘ those are Four six of local boys.',
        "City ’s 's ideal scenario but the wider Goodison spread the touchline, the more In you at of a the gem.’ chance charge: Wilcox on the net in finding fixture It Under 18 Man have Everton the you Park.",
        'of when back: Wilcox, League the during Way Blackburn 1998, Arsenal on Vieira( right), of takes, in a Premier in match colours the Rovers - 1999 season.',
        'The seeds beginning talent germinate of to are.',
        'side winners, 11s ’s Under City national Under 14s were and champions and the Under 18 Last were northern league season.',
        'Under final, UEFA Under 19 side reached the quarter- Vieira League the an Youth of.',
        'names 6 Bayern team victory on - Munich sheet around Europe, with five English A over the 0 reverberated.',
        'massive Vieira yet team youth seen in competitions first to squad, something are attributes to the‘ Most’ gap between be the and the of demands elite football.',
        'It is thought City remain why to are lower to over discussions B- teams in the receptive tiers.',
        'encapsulates into prevaricate authorities, City are single- minded little their aspiration flowing in As the journey from chip- this money caviar quite like the to and dedication and the fat academy.',
        'Time insists of: Vieira and he can stars shape effort the future.',
        'hard invested on in the securing team, first Premier two League titles in three foundations, Sheik Mansour is now Having City work heavily at ’s years.',
        'the this will, Later £ 150 m City Football Academy year open, to Tony ’s Casino from the Etihad intended, where stone build once Stadium a Blair a Super throw.',
        'the will a facilities unrivalled in It English game and - Owen, boast club We years for more do between man, to:‘ kit school reflects train on council pitches and than pre Rick season Stoke a used 20 at and Crewe.',
        'session times noticing.’ and this summer attracted, it have to breathless training How in sticky, cloying conditions On Vieira the players irritated, change that his become is become morning a has ball.',
        '‘ players, stop,’ instantly orders, his Stop freezing he.',
        '‘ at Look yourselves.',
        'together you of, all bunched Ten!',
        'like you can play How this?',
        'make for the space, Look the pitch bigger.’ He hands, out spreading his motions.',
        '‘ need understand The to boys this,’ Vieira insists.',
        '‘ When you as ball the, the pitch must big have be as possible.',
        'If you lose it, possible it as then as make to tight seven win, maximum, and seconds it back.',
        "Foreign almost: first Pellegrini team Manuel 's has been imports impossible to in into break recent years.",
        'Sergio: Star big of one is many Aguero- money moves City have made.',
        '‘ quick teams best have a The recovery.',
        'When you and, goalkeeper the is whole team, high press fast, even the it.',
        'No Manuel unbelievable— Neuer, he is like old an Watch 5.',
        'But trained was n’t born way this, he he hard.',
        'If nutshell tempo Brian, we can create the by.’ This, in a we, is these club and philosophy: an intoxicating that Marwood high- start, passing Txiki brand has been outlined players Begiristain, sporting director football Allen ’s academy director early of.',
        'It under the all that City team group at identity Manuel, from the Under-11 encourage to the first now is levels Pellegrini.',
        'Allen expands:‘ When talent took job the years four ago I outlined City 10-year see to really start to plan a of group I through coming together all playing the a way.',
        '‘ Cycles time take.',
        '90s the then In France was it, late Spain, now Germany.',
        'arrive moment will Our.’‘ The his will not cheeks,’ out concedes, puffing Vieira numbers lie.',
        '‘ We players to make a report in many on years first 10 have are in the how team.',
        'Then wrong can say we did it highlights Liverpool we did it we.’ Manchester or v VIDEO City right.']
    

if __name__ == '__main__':
    unittest.main()
