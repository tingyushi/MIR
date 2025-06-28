
import os
import numpy as np
import random
import time
import shutil
import json
import sys


from sklearn.metrics import accuracy_score

from mido import MidiFile, tempo2bpm, MidiTrack
import pretty_midi

import optuna.integration.lightgbm as lgb
from lightgbm import early_stopping
from lightgbm import log_evaluation


def extract_features(midi_obj):

    # we can use the following features
    # note duration statistics
    # velocity statistics
    # pitch interval statistics
    # Number of notes sounding at the same time(statisitcs)
    # information about percussion instruments


    durations = []
    velocities = []
    polyphony = []
    on_times = {}
    active_notes = 0
    time_counter = 0
    drum_count = 0
    melodic_count = 0
    pitch_seq = []
    tempos = []

    for track in midi_obj.tracks:
        for msg in track:

            # collect tempo information
            if msg.type == 'set_tempo':
                tempos.append(tempo2bpm(msg.tempo))

            # use for duration statistics
            time_counter += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:

                # velocity
                velocities.append(msg.velocity)

                # precussion instrument information
                if msg.channel == 9:
                    drum_count += 1
                else:
                    melodic_count += 1

                # used for note duration statistics
                on_times[msg.note] = time_counter

                # used for pitch interval statistics
                pitch_seq.append(msg.note)

                # used for polyphony statistics
                active_notes += 1

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):

                # used for note duration statistics
                if msg.note in on_times:
                    durations.append(time_counter - on_times[msg.note])
                
                # used for polyphony statistics
                active_notes = max(active_notes - 1, 0)

            polyphony.append(active_notes)

    # pitch interval information
    pitch_intervals = [abs(pitch_seq[i+1] - pitch_seq[i]) for i in range(len(pitch_seq)-1)]
    duration_intervals = [abs(durations[i+1] - durations[i]) for i in range(len(durations)-1)]
    velocities_intervals = [abs(velocities[i+1] - velocities[i]) for i in range(len(velocities)-1)]
    polyphony_intervals = [abs(polyphony[i+1] - polyphony[i]) for i in range(len(polyphony)-1)]

    # define a function to calculate statistics
    lambda_stats = lambda x: [np.mean(x), np.std(x), np.min(x), np.max(x), np.max(x) - np.min(x)] if x else np.array([0, 0, 0, 0, 0])

    # features = np.concatenate([
    #     lambda_stats(pitch_seq),
    #     lambda_stats(durations),
    #     lambda_stats(velocities),
    #     lambda_stats(polyphony),
    #     lambda_stats(pitch_intervals),
    #     [drum_count, melodic_count],
    #     lambda_stats(tempos), 
    #     [len(tempos)]
    # ])

    features = np.concatenate([
        lambda_stats(pitch_seq),
        lambda_stats(durations),
        lambda_stats(velocities),
        lambda_stats(polyphony),

        lambda_stats(pitch_intervals),
        lambda_stats(duration_intervals),
        lambda_stats(velocities_intervals),
        lambda_stats(polyphony_intervals),

        [drum_count, melodic_count],
        lambda_stats(tempos), 
        [len(tempos)]
    ])

    return features


def extract_pair_features(midi_obj1, midi_obj2):
    features1 = extract_features(midi_obj1)
    features2 = extract_features(midi_obj2)
    return np.square(features2 - features1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def pitch_shift(input_midi, semitones):
    mid = input_midi
    new_mid = MidiFile()
    
    for track in mid.tracks:
        new_track = MidiTrack()
        for msg in track:
            if msg.type in ['note_on', 'note_off']:
                new_note = msg.note + semitones
                if 0 <= new_note <= 127:
                    msg = msg.copy(note=new_note)
            new_track.append(msg)
        new_mid.tracks.append(new_track)

    return new_mid


def velocity_jitter(input_midi, jitter_value):

    mid = input_midi
    new_mid = MidiFile()
    
    for track in mid.tracks:
        new_track = MidiTrack()
        for msg in track:
            if msg.type == 'note_on':
                new_vel = max(0, min(127, msg.velocity + jitter_value))
                msg = msg.copy(velocity=new_vel)
            new_track.append(msg)
        new_mid.tracks.append(new_track)
    
    return new_mid


def scale_durations(midi_obj, duration_scale):
    """
    Scale durations of all notes in a pretty_midi.PrettyMIDI object.
    Returns a new PrettyMIDI object.
    """
    new_midi = pretty_midi.PrettyMIDI()
    
    for instrument in midi_obj.instruments:
        new_instrument = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum)
        for note in instrument.notes:
            start = note.start
            duration = (note.end - note.start) * duration_scale
            end = start + duration
            new_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=start,
                end=end
            )
            new_instrument.notes.append(new_note)
        new_midi.instruments.append(new_instrument)
    
    return new_midi


def prepare_data(dataroot, DURATION_SCALE_VALUES, SEMITONE_JITTER_VALUES, VELOCITY_JITTER_VALUES):
    
    duration_augmentation_folder_name = "duration_augmentation/"

    # precompute midi files and prettymidi objects
    with open(dataroot+"train.json", 'r') as f:
        train_json = eval(f.read())
    with open(dataroot+"test.json", 'r') as f:
        test_json = eval(f.read())
    
    allFiles = []
    for k in train_json:
        file1, file2 = k
        allFiles.append(file1)
        allFiles.append(file2)
    for k in test_json:
        file1, file2 = k
        allFiles.append(file1)
        allFiles.append(file2)
    allFiles = list(set(allFiles))

    midiStore = {}
    pmidiStore = {}
    counter = 0
    for file in allFiles:
        midiStore[file] = MidiFile( os.path.join(dataroot, file) )
        pmidiStore[file] = pretty_midi.PrettyMIDI( os.path.join(dataroot, file) )
        counter += 1
        if counter % 50 == 0:
            print(f"Loaded {counter}/{len(allFiles)} midi and pmidi files")



    # collect training data 
    training_midifile_objs = []
    training_prettymidi_objs = []
    training_labels = []
    counter = 0
    for k in train_json:
        file1, file2 = k
        midi1 = midiStore[file1]
        midi2 = midiStore[file2]
        pmidi1 = pmidiStore[file1]
        pmidi2 = pmidiStore[file2]

        training_midifile_objs.append( (midi1, midi2) )
        training_prettymidi_objs.append( (pmidi1, pmidi2) )
        training_labels.append(train_json[k])
            
    print(f"Finished loading training data into MidiFile and PrettyMIDI objects")


    augmented_mid_objs = []
    augmented_labels = []
    
    # duration augmentation using pretty_midi
    if DURATION_SCALE_VALUES:

        # remove the folder if it exists
        if os.path.exists(os.path.join(dataroot, duration_augmentation_folder_name)):
            shutil.rmtree(os.path.join(dataroot, duration_augmentation_folder_name))
        os.mkdir(os.path.join(dataroot, duration_augmentation_folder_name))

        counter = 0
        # duration augmentation
        for (pmidi1, pmidi2), label in zip( training_prettymidi_objs, training_labels):
            for duration_scale in DURATION_SCALE_VALUES:
                new_pmidi1 = scale_durations(pmidi1, duration_scale)
                new_pmidi1.write(os.path.join(dataroot, duration_augmentation_folder_name, f"duration_augmented_{counter}_.mid"))
                counter += 1

                new_pmidi2 = scale_durations(pmidi2, duration_scale)
                new_pmidi2.write(os.path.join(dataroot, duration_augmentation_folder_name, f"duration_augmented_{counter}_.mid"))
                counter += 1

                tempmidi1 = MidiFile(os.path.join(dataroot, duration_augmentation_folder_name, f"duration_augmented_{counter-2}_.mid"))
                tempmidi2 = MidiFile(os.path.join(dataroot, duration_augmentation_folder_name, f"duration_augmented_{counter-1}_.mid"))
                augmented_mid_objs.append( (tempmidi1, tempmidi2) )

                augmented_labels.append(label)

    print(f"Finished loading duration augmented data into MidiFile objects")


    # data augmentation
    for (midi1, midi2), label in zip(training_midifile_objs, training_labels):
      
        for d in SEMITONE_JITTER_VALUES:            
            newmidi1 = pitch_shift(midi1, d)
            newmidi2 = pitch_shift(midi2, d)
            augmented_mid_objs.append( (newmidi1, newmidi2) )
            augmented_labels.append(label)

        for d in VELOCITY_JITTER_VALUES:
            newmidi1 = velocity_jitter(midi1, d)
            newmidi2 = velocity_jitter(midi2, d)
            augmented_mid_objs.append( (newmidi1, newmidi2) )
            augmented_labels.append(label)
        

        # append the original midi file
        augmented_mid_objs.append((midi1, midi2))
        augmented_labels.append(label)

    print(f"Finished Pitch and Velocity Jitter data augmentation")


    # extract features
    trainX = [ extract_pair_features(midi1, midi2) for midi1, midi2 in augmented_mid_objs]
    
    # convert to numpy arrays
    trainX = np.array(trainX)
    trainY = np.array(augmented_labels)

    # prepare testing data
    with open(dataroot+"test.json", 'r') as f:
        test_json = eval(f.read())
    test_json = list(test_json)

    test_mid_objs = []
    for file1, file2 in test_json:
        midi1 = MidiFile( os.path.join(dataroot, file1) )
        midi2 = MidiFile( os.path.join(dataroot, file2) )    
        test_mid_objs.append( (midi1, midi2) )
    
    testX = np.array([extract_pair_features(midi1, midi2) for midi1, midi2 in test_mid_objs])

    return trainX, trainY, testX, test_json




if __name__ == "__main__":

    dataroot = "student_files/task2_next_sequence_prediction/"
    duration_augmentation_folder_name = "duration_augmentation/"
    
    tuningtime =  60 * 10
    config_idx = 0
    optuna_train_seed = 42
    savedir = "task2_large_features"

    getting_features = False
    training = True
    
    if config_idx == 0:
        VELOCITY_JITTER_VALUES = []
        SEMITONE_JITTER_VALUES = []
        DURATION_SCALE_VALUES = []  

    if config_idx == 1:
        VELOCITY_JITTER_VALUES = [ -2, -1, 1, 2]
        SEMITONE_JITTER_VALUES = [ -2, -1, 1, 2]
        DURATION_SCALE_VALUES = []  

    if config_idx == 2:
        VELOCITY_JITTER_VALUES = [-3, -2, -1, 1, 2, 3]
        SEMITONE_JITTER_VALUES = [-3, -2, -1, 1, 2, 3]
        DURATION_SCALE_VALUES = [] 

    if getting_features:

        config = {
            "dataroot": dataroot,
            "duration_augmentation_folder_name": duration_augmentation_folder_name,
            "VELOCITY_JITTER_VALUES": VELOCITY_JITTER_VALUES,
            "SEMITONE_JITTER_VALUES": SEMITONE_JITTER_VALUES,
            "DURATION_SCALE_VALUES": DURATION_SCALE_VALUES
        }


        tik = time.time()
        trainX, trainY, testX, test_json = prepare_data(dataroot=dataroot,
                                                        DURATION_SCALE_VALUES=DURATION_SCALE_VALUES, 
                                                        SEMITONE_JITTER_VALUES=SEMITONE_JITTER_VALUES, 
                                                        VELOCITY_JITTER_VALUES=VELOCITY_JITTER_VALUES)
        tok = time.time()


        print(f"Finish Feature Extraction using {tok-tik:.2f} seconds")
        print(f"TrainX shape: {trainX.shape}")
        print(f"TrainY shape: {trainY.shape}")
        print(f"TestX shape: {testX.shape}")


        # save a dictionary of the config as json

        with open(os.path.join(savedir ,f"config_{config_idx}.json"), 'w') as f:
            json.dump(config, f, indent=4)

        # save numpy arrays
        np.save(os.path.join(savedir, f"trainX_{config_idx}.npy"), trainX)
        np.save(os.path.join(savedir, f"trainY_{config_idx}.npy"), trainY)
        np.save(os.path.join(savedir, f"testX_{config_idx}.npy"), testX)
        np.save(os.path.join(savedir, f"test_json_{config_idx}.npy"), test_json)


    if training:

        trainX = np.load(os.path.join(savedir, f"trainX_{config_idx}.npy"))
        trainY = np.load(os.path.join(savedir, f"trainY_{config_idx}.npy"))
        testX = np.load(os.path.join(savedir, f"testX_{config_idx}.npy"))
        test_json = np.load(os.path.join(savedir, f"test_json_{config_idx}.npy"), allow_pickle=True).tolist()

        # Load or replace with your dataset
        X, y = trainX, trainY
        y = np.array(y).astype(np.uint8)

        # Prepare LightGBM Dataset (train == val to avoid data waste)
        dtrain = lgb.Dataset(X, label=y)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
        }

        tik = time.time()
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dtrain],
            callbacks=[early_stopping(100), log_evaluation(100)],
            time_budget=tuningtime,
            optuna_seed=optuna_train_seed)
        tok = time.time()

        print(f"\n********** Finish Training using {tok-tik:.2f} seconds **********\n")

        prediction = np.rint( model.predict(X, num_iteration=model.best_iteration) )
    
        accuracy = accuracy_score(y, prediction)

        print(f"\n*****************************************\n")
        best_params = model.params
        print("Best params:", best_params)
        print("  Accuracy = {}".format(accuracy))
        print("  Params: ")
        for key, value in best_params.items():
            print("    {}: {}".format(key, value))
        print(f"\n*****************************************\n")

        testprediction = np.rint( model.predict(testX, num_iteration=model.best_iteration) )
        
        prediction_results = {}
        for d, pred in zip(test_json, testprediction):
            file1 = d[0]
            file2 = d[1]
            prediction_results[(file1, file2)] = bool( pred )
        
        with open("predictions2.json", 'w') as f:
            f.write(str(prediction_results) + "\n")
            print("\nGenerate new predictions2.json\n")