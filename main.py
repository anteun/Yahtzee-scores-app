from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.properties import StringProperty, NumericProperty
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.graphics import Color, Rectangle, InstructionGroup
from kivy.properties import ListProperty
from kivy.uix.image import Image
from kivy.clock import Clock

import pandas as pd
import numpy as np
import os
import datetime
import tempfile
import time
import helper_functions
import numpy_NN

temp_dir = tempfile.TemporaryDirectory()
os.environ['MPLCONFIGDIR'] = temp_dir.name
import matplotlib.pyplot as plt

Window.softinput_mode = 'below_target'

file = 'results_yathzee.json'

score_columns = ["One", "Two", "Three", "Four", "Five", "Six", "Three of a kind", "Four of a kind", "Full House",
                 "Small straight", "Large straight", "Yathzee", "Chance"]

result_columns = ["One", "Two", "Three", "Four", "Five", "Six", "Bonus", "Three of a kind", "Four of a kind", "Full House",
                 "Small straight", "Large straight", "Yathzee", "Chance"]

keys_df = ['one_', 'two_', 'three_', 'four_', 'five_', 'six_',
         'three of a kind_', 'four of a kind_', 'full house_',
         'small straight_', 'large straight_', 'yathzee_', 'chance_',
         'total_']

path = ""
filepath = ""
pathgraphs = ""
player_names = []
database_present = False
num_games = 0

def init_app_storage(user_data_dir):
    global path, filepath, pathgraphs, player_names, database_present, num_games, players, players_added, df_import
	
    #for android
    from android.permissions import request_permissions, Permission
    from android.storage import primary_external_storage_path
    
    request_permissions([Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_EXTERNAL_STORAGE])
    
    external_storage = primary_external_storage_path()

    #path = os.path.join(os.getcwd(), 'yathzee')
    #path = os.path.join(external_storage, 'yathzee')
    path = user_data_dir
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, file)
    pathgraphs = os.path.join(path, 'Graphs.png')

    player_names = []

    #Name the player columns
    try:
        database_present = True
        df_import = pd.read_json(filepath)
        num_players = int((len(df_import.keys())-1)/14)
        players = [""] * num_players
        for i in range(num_players):
            players[i] = df_import.keys()[i*14][4:]
        players.sort()
        player_names = players
    except:
        database_present = False
        player_names = ['Player1', 'Player2']
        players = []
        num_games = 0

    num_players = len(player_names)
    players_added = num_players

class Game(FloatLayout):
    def __init__(self, **kwargs):
        super(Game, self).__init__()
        self.reload_sheet(0.9)

    def reload_sheet(self, size):
        if size == 0.9:
            size_grid = 1
        elif size == 1:
            size_grid = 0.9
        self.size_hint = (1, size)
        self.layout = GridLayout(size_hint=(1, size_grid), pos_hint={'x': 0, 'y': 0})
        self.layout.cols = len(player_names) + 1
        self.add_widget(self.layout)
        self.layout.textinputs = {}
        clear = ModernButton(text='Clear', font_name="DejaVuSans.ttf")
        clear.bind(on_release=self.clearall)
        self.layout.add_widget(clear)

        global all_keys
        all_keys = [''] * (len(score_columns) * len(player_names))

        colors = plt.cm.rainbow(np.linspace(0, 1, len(player_names)))
        #colors = plt.cm.Pastel1.colors
        self.colors_rgba = colors

        #create player headers
        excess = [0] * len(player_names)
        for a, b in enumerate(player_names):
            if len(player_names) > 1 and len(player_names) < 10:
                lbl = BarInput(
                    text=f"{b} ({excess[a]})",
                    halign="center",
                    # valign="center",
                    # color=(1, 1, 1, 1),
                    foreground_color=(1, 1, 1, 1),
                    disabled_foreground_color = self.colors_rgba[a],
                    background_color=(0, 0, 0, 0),
                    multiline=False,
                    # disabled_color=(1, 1, 1, 1),
                    colors=[self.colors_rgba[a]],
                    splits=[1],
                )
            else:
                lbl = PlayerInputs()
                lbl.text = b + ' (' + str(excess[a]) + ')'
                lbl.halign = 'center'
                lbl.valign = 'center'

            key = f"player {a}"
            self.layout.ids[key] = lbl
            self.layout.ids[key].bind(text=self.change_player)
            self.layout.add_widget(lbl)

        #create score sheet
        c = 0
        for i in score_columns:
            lbl = MyLutton()
            lbl.label_text = i
            self.layout.add_widget(lbl)
            for a, b in enumerate(player_names):
                key = i + '_' + str(a)
                all_keys[c] = str(key)
                self.layout.textinputs[key] = MyTextInput(multiline=False)
                self.layout.textinputs[key].bind(text=self.update_txt)
                if len(player_names)>1 and len(player_names)<10:
                    self.layout.textinputs[key].bind(focus=self.update_nn_probs)
                self.layout.add_widget(self.layout.textinputs[key])
                c += 1

        mybutton = ModernButton(text="Finished", font_name="DejaVuSans.ttf")
        mybutton.bind(on_press=self.total_function)
        self.layout.add_widget(mybutton)

        global keys_total
        keys_total = [0] * len(player_names)

        for a, b in enumerate(player_names):
            key = 'total' + '_' + str(a)
            keys_total[a] = str(key)
            self.layout.textinputs[key] = TotalScores()
            self.layout.add_widget(self.layout.textinputs[key])

        global all_keys_total
        all_keys_total = all_keys

        for i in range(len(player_names)):
            all_keys_total.insert(len(player_names) * 13 + i, keys_total[i])

    def update_nn_probs(self, instance, value):

        # self.shape_nn_input()
        nn_input = np.zeros((len(player_names), 2 * len(score_columns)))
        for a, b in enumerate(player_names):
            for ix, score_name in enumerate(score_columns):
                key = score_name + '_' + str(a)
                val = self.layout.textinputs[key].text

                # switch chance and yahtzee for input
                if ix == 12:
                    ix = 11
                elif ix == 11:
                    ix = 12
                # shape the nn input
                if val == '':
                    nn_input[a][ix] = 0
                    nn_input[a][ix + len(score_columns)] = 0
                else:
                    nn_input[a][ix] = int(val)
                    nn_input[a][ix + len(score_columns)] = 1

        # probabilities = np.zeros(len(player_names))
        # probability_dist = [[] for _ in range(len(player_names))]
        probs_model = numpy_NN.DynamicNumPyModel("NN_model_weights.npz")
        probability_dist = probs_model.forward(scores=nn_input[:, :13], mask=nn_input[:, 13:])
        for a, b in enumerate(player_names):
            total_points = 0
            total_points = np.sum(nn_input[a][:13])
            if np.sum(nn_input[a][:6]) >= 63:
                total_points += 35
            if np.sum(nn_input[a][13:]) == 13:
                probability_dist[a] = np.zeros(375)
                if total_points > 374:
                    total_points = 374
                probability_dist[a][int(total_points)] = 1
            elif total_points is not None or int(total_points) > 0:
                probability_dist[a] = np.concatenate([np.zeros(int(total_points)), probability_dist[a]])[
                                      :375]  # this corrects for the scores so far

        probabilities = numpy_NN.calculate_win_prob(np.array(probability_dist[:, :375]))
        # calculate splits and color_index
        # splits, color_index = self.calc_splits(probabilities)
        n = len(player_names)
        splits = [[] for _ in range(n)]
        color_index = [[] for _ in range(n)]
        # print(self.probs)
        raw_splits = probabilities * n
        # print(raw_splits)
        pos = 0.0
        for i, width in enumerate(raw_splits):
            remaining = width
            while remaining > 1e-9:
                bin_idx = min(int(pos + 1e-9), n - 1)
                room = bin_idx + 1 - pos
                piece = min(room, remaining)
                pos = round(pos + piece, 9)
                splits[bin_idx].append(round(pos - bin_idx, 9))
                color_index[bin_idx].append(i)
                remaining -= piece

        for a, b in enumerate(player_names):
            key = f"player {a}"
            self.layout.ids[key].splits = splits[a]
            self.layout.ids[key].colors = [self.colors_rgba[i] for i in color_index[a]]
            self.layout.ids[key].text_color = self.colors_rgba[a]

    def get_id(self, instance):
        for id, widget in instance.parent.ids.items():
            if widget.__self__ == instance:
                return id

    def update_txt(self, y, z):
        n_p = len(player_names)
        total_up = [0] * n_p
        total_ideal = [0] * n_p
        num = [0] * n_p * 6
        excess = [0] * len(player_names)
        ref = [i for i in range(1*3, 6*3 + 1,3) for _ in range(n_p)]
        ref3 = [0]*n_p*6

        c = 0
        for i in score_columns:
            for a in range(players_added):
                key = i + '_' + str(a)
                all_keys[c] = str(key)
                c += 1

        for b, d in enumerate(all_keys[0:n_p * 6]):
            if len(self.layout.textinputs[d].text) > 0:
                try:
                    num[b] = float(self.layout.textinputs[d].text)
                    ref3[b] = ref[b]
                except:
                    num[b] = np.nan

        for a in range(len(player_names)):
            total_up[a] = sum(num[a:n_p * 6:n_p])
            total_ideal[a] = sum(ref3[a:n_p * 6:n_p])
            excess[a] = total_up[a] - total_ideal[a]

        for a, b in enumerate(player_names):
            try:
                self.layout.ids['player ' + str(a)].text = b + ' (' + str(int(excess[a])) + ')'
            except:
                self.layout.ids['player ' + str(a)].text = b + ' (NaN)'

    def clearall(self, z):
        new = 1 + 1
        print('clearall')
        for a in all_keys_total:
            self.layout.textinputs[a].text = ''

    def total_function(self, layout):
        for a in all_keys:
            if len(self.layout.textinputs[a].text) == 0:
                self.layout.textinputs[a].text = '0'

        total = [0] * len(player_names)
        total_up = [0] * len(player_names)
        total_low = [0] * len(player_names)
        num = [0] * len(all_keys)
        for b, d in enumerate(all_keys):
            try:
                num[b] = float(self.layout.textinputs[d].text)
            except:
                num[b] = np.nan

        for a in range(len(player_names)):
            try:
                n_p = len(player_names)
                total_up[a] = sum(num[a:n_p * 6:n_p])
                total_low[a] = sum(num[6 * n_p + a:n_p * 13:n_p])
                if total_up[a] > 62:
                    total[a] = total_up[a] + total_low[a] + 35
                else:
                    total[a] = total_up[a] + total_low[a]
            except:
                total[a] = np.nan

        keys_total = [0] * len(player_names)
        for a, b in enumerate(player_names):
            key = 'total' + '_' + str(a)
            keys_total[a] = str(key)
            if np.isnan(total).any():
                self.layout.textinputs[keys_total[a]].text = 'NaN'
                stop_save = True
            else:
                stop_save = False
                self.layout.textinputs[keys_total[a]].text = str(int(total[a]))

        if stop_save:
            return

        winner_index = np.flatnonzero(total == np.nanmax(total))
        if len(winner_index)==1:
            winner = player_names[winner_index[0]] + ' wins!'
        else:
            winner = ''
            for i in winner_index:
                winner += player_names[i] + ' & '
            winner = winner[:-3]
            winner += ' draw!'

        results = winner
        for i,player in enumerate(player_names):
                results += '\n' + player + ': ' + str(round(total[i]))

        self.box = BoxLayout(orientation='horizontal', size_hint=(1, 1))
        self.lbl = Label(text=results)
        self.butclose = ModernButton(text="Ok", size_hint=(0.3, 0.3), pos_hint={'x': 0.8, 'y': 0.1}, font_name="DejaVuSans.ttf")
        self.box.add_widget(self.lbl)
        self.box.add_widget(self.butclose)

        self.win_pop = Popup(title="Results", content=self.box, size_hint=(0.5, 0.25), auto_dismiss=False)

        self.butclose.bind(on_press=self.win_pop.dismiss)
        self.butclose.bind(on_press=self.show_it)

        self.win_pop.open()

    def show_it(self, root):
        self.box = BoxLayout(orientation='horizontal', size_hint=(1, 1))
        self.butyes = ModernButton(text="Yes", size_hint=(0.5, 0.8), font_name="DejaVuSans.ttf")
        self.butno = ModernButton(text="No", size_hint=(0.5, 0.8), font_name="DejaVuSans.ttf")
        self.box.add_widget(self.butyes)
        self.box.add_widget(self.butno)

        self.main_pop = Popup(title="Save game?", content=self.box, size_hint=(0.5, 0.25), auto_dismiss=False)

        self.butyes.bind(on_press=self.main_pop.dismiss)
        self.butyes.bind(on_press=self.save_function)
        self.butno.bind(on_press=self.main_pop.dismiss)

        self.main_pop.open()

    def players_in_database(self, df_import, player_names):
        players = []

        # playernames in lowercase
        for a in player_names:
            players.append(a.lower())

        # select for only players that are in database
        players_database = []
        for i in range(int((len(df_import.keys())-1)/14)):
            b = i * 14
            current_player = df_import.keys()[b][4:]

            players_database.append(current_player.lower())

        #select current players that are also present in the database
        gameplayers_in_database = [s for s in players if s in players_database]

        players_database.sort()
        gameplayers_in_database.sort()

        return players_database, gameplayers_in_database

    def import_data(self, players, keys_df, filepath):
        no_games_yet = True
        df_import = None

        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            try:
                df_import = pd.read_json(filepath)
                no_games_yet = False  
            except Exception as e:
                print(f"No correct .json file found or corrupt file: {e}", flush=True)
                no_games_yet = True

        if no_games_yet:
            new_keys_database = [f"{key}{player}" for player in players for key in keys_df]
            new_keys_database.append('date')

            fallback_dict = {col: [] for col in new_keys_database}
            df_import = pd.DataFrame(fallback_dict)

        return df_import

    def save_function(self, root):
        players = []
        # playernames in lowercase
        for a in player_names:
            players.append(a.lower())

        # find data base, if absent create one
        df_import = self.import_data(players, keys_df, filepath)

        # select for only players that are in database
        players_database, gameplayers_in_database = self.players_in_database(df_import, player_names)

        # select scores and save it
        n_p = len(players_database)
        num = [0] * 14
        keys_save = [0] * 14 * n_p
        if gameplayers_in_database == players_database:
            df_overall = []
            for current_player in players_database:
                i = players.index(current_player)
                keys_save = all_keys_total[i:len(players) * 14:len(players)]
                for b, d in enumerate(keys_save):
                    num[b] = float(self.layout.textinputs[d].text)
                df_overall += num

            game = ['Game ' + str(len(df_import) + 1)]
            idx = df_import.keys().tolist()
            idx_new = [""] * len (idx)

            date = [datetime.datetime.now()]
            df_overall = df_overall + date
            dummy = pd.DataFrame(df_overall, index=idx, columns=game).T
            df_import = pd.concat([df_import, dummy]).fillna(0)
            df_import.to_json(filepath)
            print(filepath)
            print('Results Saved')
            global database_present
            database_present = True
        else:
            gameplayers_not_in_database = [s for s in players_database if s not in players]
            self.box = BoxLayout(orientation='horizontal', size_hint=(1, 1))
            self.lbl = Label(text=f'{gameplayers_not_in_database[:]} did not join, \n results not saved')
            self.butclose = ModernButton(text="Ok", size_hint=(0.3, 0.3), pos_hint={'x': 0.8, 'y': 0.1},
                                   font_name="DejaVuSans.ttf")
            self.box.add_widget(self.lbl)
            self.box.add_widget(self.butclose)
            self.err_pop = Popup(title="Player not found", content=self.box, size_hint=(0.5, 0.25), auto_dismiss=False)
            self.butclose.bind(on_press=self.err_pop.dismiss)
            self.err_pop.open()

            print(f'{gameplayers_not_in_database} did not join, results not saved')

    def change_player(self, y, z):
        global player_names
        for a, b in enumerate(player_names):
            lbl_text = self.layout.ids['player ' + str(a)].text
            if " " in lbl_text:
                player_names[a] = lbl_text[:lbl_text.index(" ")]
            elif "(" in lbl_text:
                player_names[a] = lbl_text[:lbl_text.index("(")]

    def remove_player(self):
        global players_added
        global player_names
        if players_added > 1:
            players_added -= 1
            player_names = player_names[0:players_added]
            self.reload_sheet(1)

    def add_player(self):
        global players_added
        global player_names
        if len(player_names)<10:
            players_added += 1
            player_names.insert(players_added - 1, 'Player' + str(players_added))
            self.reload_sheet(1)
    pass

class Statistics(BoxLayout):
    def __init__(self, **kwargs):
        super(Statistics, self).__init__()
        df_import = self.stats_import(players, keys_df, filepath)

        if database_present:
            player_stats, txt, wins_ordered, player_totals, win_sum_player, bonus_player = self.stats_calc(df_import, keys_df)
        else:
            player_stats = {}
            player_totals = {}
            win_sum_player = {}
            bonus_player = {}
            for i, player in enumerate(player_names):
                player_stats[player] = [player, '0', '0', '0', '0', '0', '0', '0']
                txt = ['Games: 0', 'Wins:', 'Mean:', 'Median:', 'Max:', 'Min:', 'Yathzee:', 'Winstreak:', 'Bonus:']
                wins_ordered = []
                player_totals[player] = []
                win_sum_player[player] = []
                bonus_player[player] = []

        self.stats_layout(player_stats, txt, wins_ordered, player_totals, win_sum_player, bonus_player, df_import)

    def stats_import(self, players, keys_df, filepath):
        df_import = Game.import_data(self, players, keys_df, filepath)
        return df_import

    def count_winlossstreak(self, player_num, wins_ordered_player):
        answin = 0
        ansloss = 0
        countwin = 0
        countloss = 0
        win_sum = np.zeros(len(wins_ordered_player))

        for i, v in enumerate(wins_ordered_player):
            if v==-1:
                countwin = 0
                countloss = 0
                win_sum[i] = win_sum[i - 1]
            elif v == player_num:
                win_sum[i] = win_sum[i - 1] + 1
                if v == wins_ordered_player[i - 1]:
                    countwin += 1
                else:
                    countwin = 1
            elif v != player_num:
                win_sum[i] = win_sum[i - 1]
                if v == wins_ordered_player[i - 1]:
                    countloss += 1
                else:
                    countloss = 1


            answin = max(answin, countwin)
            ansloss = max(ansloss, countloss)

        return answin, ansloss, win_sum

    def stats_calc(self, df_import, keys_df):
        if not database_present:
            return
        player_totals = {}
        player_yathzee = {}

        players = []
        # playernames in lowercase
        for a in player_names:
            players.append(a.lower())
        # select for only players that are in database
        players_database, gameplayers_in_database = Game.players_in_database(self, df_import, players)

        for i, player in enumerate(players_database):
            all_keys = {}

            # backwards compatibility with old database
            for key in keys_df:
                all_keys[key + player] = key + player
            player_totals[player] = df_import[all_keys[f'total_{player}']].fillna(0).tolist()

            player_yathzee[player] = sum([x / 50 for x in df_import[all_keys[f'yathzee_{player}']].fillna(0).tolist()])

        wins_ordered = [np.nan] * len(player_totals[players_database[0]])

        # win order
        for i in range(len(wins_ordered)):
            scores_temp = []
            for player in players_database:
                scores_temp.append(player_totals[player][i])

            winner = np.flatnonzero(scores_temp == np.nanmax(scores_temp))
            if len(winner) > 1:
                winner = [-1]
            winner = winner[0]

            wins_ordered[i] = winner

        winstreak_player = {}
        lossstreak_player = {}
        win_sum_player = {}

        for i, player in enumerate(players_database):
            winstreak_player[player], lossstreak_player[player], win_sum_player[player] = self.count_winlossstreak(i, wins_ordered)

        #check if bonus was obtained
        bonus_player = {}
        for i, player in enumerate(players_database):
            bonus_player[player] = 0
            for a in range(len(wins_ordered)):
                score=0
                keys_now = df_import.keys()[i*14:(i+1)*14-8]

                for k in keys_now:
                    score += df_import[k][a]
                if score>62:
                    bonus_player[player] += 1

        player_stats = {}

        for player_num, player in enumerate(players_database):
            player_stats[player] = [player, int(win_sum_player[player][-1]),
                                        "{:.1f}".format(np.mean(player_totals[player])),
                                        "{:.1f}".format(np.median(player_totals[player])),
                                        str(int(np.max(player_totals[player]))),
                                        str(int(np.min(player_totals[player]))),
                                        str(int(player_yathzee[player])),
                                        str(winstreak_player[player]), str(bonus_player[player])]

        txt = ['Games: ' + str(len(player_totals[player])), 'Wins:', 'Mean:', 'Median:', 'Max:', 'Min:', 'Yathzee:',
               'Winstreak:', 'Bonus:']

        return player_stats, txt, wins_ordered, player_totals, win_sum_player, bonus_player

    def stats_layout(self, player_stats, txt, wins_ordered, player_totals, win_sum_player, bonus_player, df_import):
        if not database_present:
            return
        players = []
        # playernames in lowercase
        for a in player_names:
            players.append(a.lower())
        # select for only players that are in database
        players_database, gameplayers_in_database = Game.players_in_database(self, df_import, players)

        gridlayout = GridLayout(size_hint=(1, 0.3), pos_hint={'x': 0, 'y': 0.6})
        gridlayout.cols = len(players_database)+1
        self.add_widget(gridlayout)

        for a, b in enumerate(txt):
            lbl = MyLutton()
            lbl.label_text = b
            gridlayout.add_widget(lbl)
            for i, player in enumerate(players_database):
                lbl = TotalScores()
                lbl.text = str(player_stats[player][a])
                gridlayout.add_widget(lbl)

        # plot data
        max_lim = 0
        min_lim = 0
        if len(players_database) == 2:
            colors = ['salmon', 'lightskyblue']
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, len(players_database)))

        plt.style.use('dark_background')
        plt.rcParams.update({'font.size': 16})
        plt.style.use('dark_background')
        plt.rcParams.update({'font.size': 16})
        fig, ax1 = plt.subplots(2, figsize=(12, 10))
        if len(players_database) == 2:
            balance_wins = win_sum_player[players_database[0]] - win_sum_player[players_database[1]]
            try:
                num_games = len(balance_wins)
                if max(balance_wins) > max_lim:
                    max_lim = max(balance_wins)
                if min(balance_wins) < min_lim:
                    min_lim = min(balance_wins)
            except ValueError:
                max_lim = 1
                min_lim = -1
                num_games = 0
            if min_lim == 0:
                min_lim = -1
            if max_lim == 0:
                max_lim = 1

            ax1[0].plot(balance_wins, linewidth=2.0, color="white")
            ax1[0].set(ylabel='Win balance', ylim=[float(min_lim)*1.05, float(max_lim)*1.05])
            ax1[0].text(0.1, 0.9, players_database[0],
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax1[0].transAxes)
            ax1[0].text(0.1, 0.1, players_database[1],
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax1[0].transAxes)
        else:
            for i, player in enumerate(players_database):
                ax1[0].plot(list(range(1, len(win_sum_player[player]) + 1)), win_sum_player[player], linewidth=2.0, color=colors[i])
                try:
                    num_games = len(win_sum_player[player])
                    if max(win_sum_player[player]) > max_lim:
                        max_lim = max(win_sum_player[player])
                except ValueError:
                    max_lim = 1
                    num_games = 0
            if max_lim == 0:
                max_lim = 1
            ax1[0].set(ylabel='Win order', ylim=[float(0), float(max_lim)])

        #bins = np.linspace(100, 350, 11)
        bins = np.arange(75, 376, 5)
        bins = np.insert(bins, 0, 75) #underflow bin
        bins = np.append(bins, 375) #overflow bin

        x_ticks = [75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375]
        x_tick_labels = ([""]+[f"≤100"]+ [f"{int(x)}" for x in x_ticks[2:-2]]+ [f"≥350"]+ [""])

        hist_totals = np.zeros((num_games, len(players_database)))

        #calculate win percentage histogram
        totals_lists = []
        for i,p in enumerate(players_database):
            totals_lists.append(player_totals[p])

        winner = np.zeros((len(players_database), len(player_totals[p])))
        for i,values in enumerate(zip(*totals_lists)):
            win_ix = values.index(max(values))
            if values.count(values[win_ix]) == 1:
                winner[win_ix][i] = 1

        bin_centers = []
        win_percentage = []
        smooth_win_percentage = []

        for i_p, player in enumerate(players_database):
            bin_index = np.digitize(np.clip(player_totals[player], a_min=99, a_max=351), bins) - 1

            bin_centers.append([])
            win_percentage.append([])

            for i in range(len(bins) - 1):
                mask = bin_index == i
                if np.any(mask):
                    bin_centers[i_p].append((bins[i] + bins[i + 1]) / 2)
                    win_percentage[i_p].append(100 * winner[i_p][mask].mean())

        #
        for i, player in enumerate(players_database):
            hist_totals[:, i] = np.clip(player_totals[player], a_min=99, a_max=351)  # set min and max value for plotting
            counts, edges = np.histogram(hist_totals[:, i], bins=bins)
            max_counts = np.max(counts)
            if np.max(counts) > max_counts:
                max_counts = np.max(counts)

        for i, player in enumerate(players_database):
            hist_totals[:, i] = np.clip(player_totals[player], a_min=99, a_max=351) #set min and max value for plotting
            counts, edges = np.histogram(hist_totals[:,i], bins=bins)

            # Compute bin centers
            centers = (edges[:-1] + edges[1:]) / 2
            x_smooth = np.linspace(centers.min(), centers.max(), 1000)
            y_smooth = helper_functions.cubic_spline_interpolate(x_smooth, centers, counts/max_counts)

            bin_centers_smooth = np.linspace(np.min(bin_centers[i]), np.max(bin_centers[i]),1000)
            try:
                win_smooth = helper_functions.cubic_spline_interpolate(bin_centers_smooth, bin_centers[i], np.array(win_percentage[i])/100)
                if num_games>49:
                    smooth_plot = True
                else:
                    smooth_plot = False
            except:
                smooth_plot = False

            if smooth_plot:
                ax1[1].plot(x_smooth, y_smooth, linewidth=2, color=colors[i], label='Scores ' + player)
                ax1[1].plot(bin_centers[i], np.array(win_percentage[i])/100, ':', linewidth = 2, color = colors[i], label = 'Win probability '+player)
            else:
                ax1[1].plot(centers, counts/max_counts, linewidth=2, color=colors[i], label='Scores ' + player)

        #ax1[1].hist(hist_totals, bins=bins, label=players_database, color=colors)
        ax1[1].legend(loc='best', facecolor= 'none', edgecolor='none')
        ax1[1].set(xlabel='Score', ylabel='Frequency', xticks = x_ticks, xticklabels=x_tick_labels)
        ticklines = ax1[1].get_xticklines()  # returns a list of Line2D objects
        ticklines[0].set_visible(False)
        ticklines[-1].set_visible(False)
        ax1[1].margins(x=0.01)
        plt.savefig(pathgraphs, transparent=True)

        box = BoxLayout(orientation='vertical')
        box.id = 'stats_box'
        box.size_hint = (1, 0.6)
        box.pos_hint = {'x': 0, 'y': 0}
        self.add_widget(box)
        imag1 = Image(source=pathgraphs)
        imag1.reload()
        box.add_widget(imag1)
    pass

class Statistics2(BoxLayout):
    def __init__(self, **kwargs):
        super(Statistics2, self).__init__()
        self.reload_stats2(result_columns=result_columns)


    def reload_stats2(self, result_columns):
        df_import = Statistics.stats_import(self, players, keys_df, filepath)

        players_database, gameplayers_in_database = Game.players_in_database(self,df_import, player_names)

        self.size_hint = (1, 1)
        self.layout = GridLayout(size_hint=(1, 0.8), pos_hint={'x': 0, 'y': 0.1})
        self.layout.cols = len(players_database) + 1
        self.add_widget(self.layout)
        self.layout.totals = {}

        #create empty cell
        lbl = TotalScores()
        self.layout.ids['empty'] = lbl
        self.layout.add_widget(lbl)

        #create player headers
        for a, player in enumerate(players_database):
            lbl = TotalScores(markup=True)
            lbl.text = player
            self.layout.ids[player] = lbl
            self.layout.add_widget(lbl)

        #create score sheet
        c = 0
        for i in result_columns:
            lbl = Button(font_name="DejaVuSans.ttf")
            lbl.bind(on_press=self.graph_rows)
            lbl.text = i
            self.layout.add_widget(lbl)
            for a, player in enumerate(players_database):
                key = i.lower() + '_' + player
                self.layout.totals[key] = TotalScores(markup=True)
                self.layout.totals[key].text = self.add_means(df_import, key, player)
                self.layout.add_widget(self.layout.totals[key])
                c += 1

        # create empty cell
        lbl = Button()
        lbl.bind(on_press=self.graph_rows)
        lbl.text = 'Total'
        self.layout.ids['Total'] = lbl
        self.layout.add_widget(lbl)

        for a, player in enumerate(players_database):
            key = 'total'
            key += '_' + player

            self.layout.totals[key] = TotalScores(markup=True)
            self.layout.totals[key].text = self.add_means(df_import, key, player)
            self.layout.add_widget(self.layout.totals[key])

        box = BoxLayout(orientation='vertical')
        box.id = 'stats2_box'
        box.size_hint = (1, 0.1)
        box.pos_hint = {'x': 0, 'y': 0}
        self.add_widget(box)
        label = Label(halign='center', valign='middle', markup=True, font_name='DejaVuSans.ttf',
            text= "[color=00ff00]▲[/color] Increase in average over last 10 games  \n[color=ff0000]▼[/color] Decrease in average over last 10 games")
        box.add_widget(label)

    def add_means(self, df_import, key, player):
        result_txt = ""
        overall_mean = 0
        last_mean = 0

        if key[:5].lower() != 'bonus':
            overall_mean = float(round(np.mean(df_import[key]), 1))
            key_temp = key
        else:
            key_temp = keys_df[0]+player

        if len(df_import[key_temp]) < 10:
            game_averaging = 1
            last_mean = float(round(df_import[key_temp][-game_averaging:], 1))
        elif len(df_import[key_temp]) < 100:
            game_averaging = 5
            last_mean = float(round(np.mean(df_import[key_temp][-game_averaging:]), 1))
        else:
            game_averaging = 10
            last_mean = float(round(np.mean(df_import[key_temp][-game_averaging:]), 1))

        score_columns_lower = [s.lower() for s in score_columns[6:-1]]
        percent_score_columns_lower = score_columns_lower[6:-1]

        result_txt = f"{overall_mean}"
        change_mean = str(round(abs(overall_mean - last_mean), 1))

        if key[:11] == keys_df[8]:
            overall_mean = round(overall_mean / 25 * 100, 1)
            last_mean = round(last_mean / 25 *100,1)
            result_txt = f"{overall_mean}"
            change_mean = f'{round(abs(overall_mean - last_mean), 1)} %'
        elif key[:14] == keys_df[9]:
            overall_mean = round(overall_mean / 30 * 100, 1)
            last_mean = round(last_mean / 30 *100,1)
            result_txt = f"{overall_mean}"
            change_mean = f'{round(abs(overall_mean - last_mean), 1)} %'
        elif key[:13] == keys_df[10]:
            overall_mean = round(overall_mean / 40 * 100, 1)
            last_mean = round(last_mean / 40 * 100, 1)
            result_txt = f"{overall_mean}"
            change_mean = f'{round(abs(overall_mean - last_mean), 1)} %'
        elif key[:8] == keys_df[11]:
            overall_mean = round(overall_mean/50 * 100,1)
            last_mean = round(last_mean / 50 * 100, 1)
            result_txt = f"{overall_mean}"
            change_mean = f'{round(abs(overall_mean - last_mean), 1)} %'
        elif key[:5].lower() == 'bonus':
            bonus_list = df_import[keys_df[-1]+player] - np.sum([df_import[key+player] for key in keys_df[:-1]],axis=0)
            overall_mean = round(np.mean(bonus_list)/35*100,1)
            last_mean = round(np.mean(bonus_list[-game_averaging:])/35*100,1)
            result_txt = f"{overall_mean}"
            change_mean = f'{round(abs(overall_mean - last_mean), 1)} %'

        if last_mean < overall_mean:
            result_txt += " [color=ff0000]▼[/color]" + change_mean
        elif overall_mean == last_mean:
            pass
        else:
            result_txt += " [color=00ff00]▲[/color]" + change_mean
        return result_txt

    def graph_rows(self, button):
        if not database_present:
            return
        players = []
        # playernames in lowercase
        for a in player_names:
            players.append(a.lower())

        df_import = Game.import_data(self, players, keys_df, filepath)

        # select for only players that are in database
        players_database, gameplayers_in_database = Game.players_in_database(self, df_import, players)

        if len(players_database) == 2:
            colors = ['salmon', 'lightskyblue']
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, len(player_names)))

        plt.style.use('dark_background')
        plt.rcParams.update({'font.size': 16})
        plt.style.use('dark_background')
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(1, figsize=(12, 10))

        for i, player in enumerate(players_database):
            use_percent_axis = False
            key = button.text.lower()
            if key == 'total':
                key='total'

            key += '_'+player
            player_temp = player

            if key[:5]!='bonus':
                data = df_import[key]
            else:
                data = (df_import[keys_df[-1]+player_temp] - np.sum([df_import[key+player_temp] for key in keys_df[:-1]],axis=0))/35*100
                use_percent_axis = True

            #select averaging conditions
            if len(data)<20:
                x_data_smallbin = list(range(0,len(data)))
                x_data_largebin = list(range(0, len(data)))
                y_data_smallbin = data
                y_data_largebin = data
            elif len(data)<100:
                game_averaging_small = 5
                game_averaging_large = 20
                x_data_smallbin = np.array(list(range(0, len(data), game_averaging_small)))+game_averaging_small/2
                x_data_largebin = np.array(list(range(0, len(data), game_averaging_large)))+game_averaging_large/2
                y_data_smallbin = np.array([np.mean(data[i:i + game_averaging_small]) for i in range(0, len(data), game_averaging_small)])
                y_data_largebin = np.array([np.mean(data[i:i + game_averaging_large]) for i in range(0, len(data), game_averaging_large)])

            else:
                game_averaging_small = round(len(data) / 100)
                game_averaging_large = round(len(data)/20)
                x_data_smallbin = np.array(list(range(0, len(data), game_averaging_small)))+game_averaging_small/2
                x_data_largebin = np.array(list(range(0, len(data), game_averaging_large)))+game_averaging_large/2
                y_data_smallbin = np.array([np.mean(data[i:i + game_averaging_small]) for i in range(0, len(data), game_averaging_small)])
                y_data_largebin = np.array([np.mean(data[i:i + game_averaging_large]) for i in range(0, len(data), game_averaging_large)])

            #calculate percentages
            if key[:11] == keys_df[8]:
                y_data_smallbin = y_data_smallbin/25*100
                y_data_largebin = y_data_largebin / 25 * 100
                use_percent_axis = True
            elif key[:14] == keys_df[9]:
                y_data_smallbin = y_data_smallbin / 30 * 100
                y_data_largebin = y_data_largebin / 30 * 100
                use_percent_axis = True
            elif key[:13] == keys_df[10]:
                y_data_smallbin = y_data_smallbin / 40 * 100
                y_data_largebin = y_data_largebin / 40 * 100
                use_percent_axis = True
            elif key[:8] == keys_df[11]:
                y_data_smallbin = y_data_smallbin / 50 * 100
                y_data_largebin = y_data_largebin / 50 * 100
                use_percent_axis = True

            x_smooth = np.linspace(min(x_data_largebin), max(x_data_largebin), 1000)
            try:
                y_smooth = helper_functions.cubic_spline_interpolate(x_smooth, x_data_largebin, y_data_largebin)
                if num_games>19:
                    smooth_plot = True
                else:
                    smooth_plot = False
            except:
                smooth_plot = False

            ax.plot(x_data_smallbin, y_data_smallbin, linewidth=2, color=colors[i], label=player)
            if smooth_plot:
                ax.plot(x_smooth,y_smooth,'--',linewidth=4,color=colors[i],label=player)

        ax.set_xlabel('Game number')
        score_name = key[:-(len(player) + 1)]

        if use_percent_axis:
            ax.set_ylabel(f'Average {score_name} (%)')
        else:
            ax.set_ylabel(f'Average {score_name}')
        ax.legend(loc='best', facecolor= 'none', edgecolor='none')
        saveloc_details = f'{path}/{score_name}_graph.png'
        fig.savefig(saveloc_details, transparent=True)

        popup_box = BoxLayout(orientation='vertical')
        img = Image(source=saveloc_details)
        img.reload()
        popup_box.add_widget(img)

        self.stats_pop = Popup(
            title="Score overview",
            content=popup_box,
            size_hint=(1, 0.5),
            auto_dismiss=True)

        self.stats_pop.open()

        try:
            os.remove(saveloc_details)
            print(f"File '{saveloc_details}' deleted successfully.")
        except:
            FileNotFoundError: print(f"File '{saveloc_details}' not found.")

class StartScreen(Screen):
    pass

class GameScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        probs_model = numpy_NN.DynamicNumPyModel("NN_model_weights.npz")

    def remove_player(self):
        Game.remove_player(self)

    def add_player(self):
        Game.add_player(self)

    def change_player(self, y, z):
        Game.change_player(self, y, z)

    def players_in_database(self, df_import, players):
        players_database, gameplayers_in_database = Game.players_in_database(self, df_import, players)
        return players_database, gameplayers_in_database

    def import_data(self, players, keys_df, filepath):
        df_import = Game.import_data(self, players, keys_df, filepath)
        return df_import

    def reload_sheet(self, size):
        Game.reload_sheet(self, size)

    def update_nn_probs(self, instance, value):
        Game.update_nn_probs(self, instance, value)

    def clearall(self, z):
        Game.clearall(self, z)

    def save_function(self, root):
        Game.save_function(self, root)

    def show_it(self, root):
        Game.show_it(self, root)

    def total_function(self, layout):
        Game.total_function(self, layout)

    def update_txt(self, y, z):
        Game.update_txt(self, y, z)

    def get_id(self):
        Game.get_id(self)

    pass

class StatsScreen(Screen):
    def stats_add(self):
        if not database_present:
            return
        self.df_import = Statistics.stats_import(self, players, keys_df, filepath)
        player_stats, txt, wins_ordered, player_totals, win_sum_player, bonus_player = Statistics.stats_calc(self, self.df_import, keys_df)
        Statistics.stats_layout(self, player_stats, txt, wins_ordered, player_totals, win_sum_player, bonus_player, self.df_import)

    def count_winlossstreak(self, player_num, wins_ordered_player):
        if not database_present:
            return
        answin, ansloss, win_sum = Statistics.count_winlossstreak(self, player_num, wins_ordered_player)
        return answin, ansloss, win_sum
    pass

class StatsScreen2(Screen):
    def reload_stats2(self):
        if not database_present:
            return
        Statistics2.reload_stats2(self, result_columns)

    def add_means(self, df_import, key, player):
        if not database_present:
            return
        self.result_txt = Statistics2.add_means(self, df_import, key, player)
        return self.result_txt

    def graph_rows(self, button):
        if not database_present:
            return
        Statistics2.graph_rows(self, button)
    pass

class SettingsScreen(Screen, Label):

    saveloc = filepath

    def confirm_pop(self):
        self.box = BoxLayout(orientation='horizontal', size_hint=(1, 1))
        self.butyes = ModernButton(text="Yes", size_hint=(0.5, 0.8), font_name="DejaVuSans.ttf")
        self.butno = ModernButton(text="No", size_hint=(0.5, 0.8), font_name="DejaVuSans.ttf")
        self.box.add_widget(self.butyes)
        self.box.add_widget(self.butno)

        self.main_pop = Popup(title="Delete game?", content=self.box, size_hint=(0.5, 0.25), auto_dismiss=False)

        self.butyes.bind(on_press=self.main_pop.dismiss)
        self.butyes.bind(on_press=self.remove)
        self.butno.bind(on_press=self.main_pop.dismiss)

        self.main_pop.open()

    def add_lbl(self):
        self.ids.lbl = Label()
        global gamenum
        self.ids.lbl.text = "Removed " + gamenum[0]
        self.ids.lbl.size_hint = (0.5, 0.25)
        self.ids.lbl.pos_hint = {'x': 0.5, 'y': 0.65}
        self.add_widget(self.ids.lbl)
        Clock.schedule_once(self.lbl_end, 4)

    def lbl_end(self, root):
        self.ids.lbl.text = ""

    def remove(self, root):
        if not database_present:
            return
        df_import = pd.read_json(filepath)
        global gamenum
        gamenum = ['Game ' + str(len(df_import))]
        df_import.drop(gamenum, inplace=True)
        df_import.to_json(filepath)
        print('Removed: ' + gamenum[0])
        self.add_lbl()
    pass

class MyTextInput(TextInput):
    input_filter = "int"
    input_type = "number"
    pass

class MyLutton(Button):
    label_text = StringProperty()
    label_id = StringProperty()

class ModernButton(Button):
    border_radius = NumericProperty(0)
    pass

class PlayerInputs(TextInput):
    def on_parent(self, widget, parent):
        self.input_type = 'text'
    pass

class BarInput(PlayerInputs):
    splits = ListProperty([])
    colors = ListProperty([])
    text_color = ListProperty([])

    def __init__(self, **kw):
        #kw.setdefault("color", (0, 0, 0, 0))  # Fully transparent text by default
        kw.setdefault("background_color", (0,0,0,0))
        #kw.setdefault("foreground_color", (1,1,1,1))
        kw.setdefault("multiline", False)

        super().__init__(**kw)

        self._group = InstructionGroup()
        self.canvas.before.add(self._group)

        self.bind(
            pos=self.redraw,
            size=self.redraw,
            colors=self.redraw,
            splits=self.redraw,
            text_color = self.redraw,
        )

    def redraw(self, *a):
        self._group.clear()

        # Guard against zero/uninitialized dimensions during layout passes
        if self.width <= 1 or self.height <= 1:
            return

        # Sanitize internal splits: keep floats strictly between 0 and 1
        internal_splits = sorted([float(s) for s in self.splits if 0 < s < 1.0])
        edges = [0.0] + internal_splits + [1.0]

        # Draw each colored segment
        for i, c in enumerate(self.colors):
            if i + 1 >= len(edges):
                break

            x_start = self.x + (edges[i] * self.width)
            segment_width = (edges[i + 1] - edges[i]) * self.width

            # Skip drawing zero-width or negative segments
            if segment_width <= 0:
                continue

            color_args = [float(x) for x in c]

            self._group.add(Color(*color_args))
            self._group.add(
                Rectangle(
                    pos=(x_start, self.y),
                    size=(segment_width, self.height/6),
                )
            )
        if len(self.text_color)>1:
            self._group.add(Color(*self.text_color))

class TotalScores(Label):
    pass

class Yathzee(App):
    Window.clearcolor = (66/256, 66/256, 66/256, 1)
    def build(self):
        init_app_storage(self.user_data_dir)

        sm = ScreenManager()
        sm.add_widget(StartScreen(name='menu'))
        sm.add_widget(GameScreen(name='game'))
        sm.add_widget(StatsScreen(name='stats'))
        sm.add_widget(StatsScreen2(name='stats2'))
        sm.add_widget(SettingsScreen(name='settings'))
        return sm


Yathzee().run()
