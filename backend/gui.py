import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Suppress TF logs matching run_clean.sh
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'



from predictive_ai.xgboost_model import XGBoostModel
from predictive_ai.lstm_model import LSTMModel
from predictive_ai.rnn_model import RNNModel
from generative_ai.dcgan_model import DCGAN, DCGANClassifier
from generative_ai.cgan_model import CGAN, CGANClassifier
from generative_ai.gan_model import GAN, GANClassifier
from generative_ai.vae_model import VAEClassifier
from generative_ai.ctgan_model import CTGANModel
from generative_ai.diffusion_model import TabDDPM

datasets = ['iris', 'heart', 'breast', 'wine', 'phishing', 'mushroom', 'gendername']
models = ['lgbm', 'xgb', 'dcgan', 'cgan', 'ctgan', 'vae', 'diffusion', 'rnn', 'lstm', 'gan']

st.title('ML Model Suite GUI')

st.sidebar.header('Configuration')
selected_dataset = st.sidebar.selectbox('Dataset', datasets)
selected_models = st.sidebar.multiselect('Models', models, default=['lgbm', 'xgb', 'dcgan'])

if st.sidebar.button('Run Selected Models', type='primary'):
    with st.spinner('Training models...'):
        progress_bar = st.progress(0)
        results = {'dataset': selected_dataset}
        
        for i, model_name in enumerate(selected_models):
            with st.status(f'Training {model_name.upper()}...'):
                try:
                    model_start = time.time()
                    
                    X_train, X_test, y_train, y_test, scaler, encoder, is_binary = load_and_preprocess_dataset(selected_dataset)
                    input_dim = X_train.shape[1]
                    num_classes = 2 if is_binary else y_train.shape[1]
                    
                    acc = 0.0
                    
                    X_rnn_train, X_rnn_test = reshape_for_rnn(X_train, X_test)
                    
                    if model_name == 'lgbm':
                        model = LightGBMModel()
                        model.build(input_dim, num_classes, is_binary)
                        model.train(X_train, y_train)
                        acc = model.evaluate(X_test, y_test)['accuracy']
                    elif model_name == 'xgb':
                        model = XGBoostModel()
                        model.build(input_dim, num_classes, is_binary)
                        model.train(X_train, y_train)
                        acc = model.evaluate(X_test, y_test)['accuracy']
                    elif model_name == 'rnn':
                        model = RNNModel(hidden_units=50, dropout=0.3)
                        model.build((1, input_dim), num_classes, is_binary)
                        model.train(X_rnn_train, y_train, epochs=30, verbose=0)
                        acc = model.evaluate(X_rnn_test, y_test)['accuracy']
                    elif model_name == 'lstm':
                        model = LSTMModel(hidden_units=50, dropout=0.1)
                        model.build((1, input_dim), num_classes, is_binary)
                        model.train(X_rnn_train, y_train, epochs=30, verbose=0)
                        acc = model.evaluate(X_rnn_test, y_test)['accuracy']
                    elif model_name == 'gan':
                        gan = GAN(input_dim)
                        gan.build_generator()
                        gan.build_discriminator()
                        gan.build_gan()
                        gan.train(X_train)
                        classifier = GANClassifier()
                        classifier.build(input_dim, num_classes)
                        y_train_sparse = np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train.ravel()
                        generated = gan.generate_samples(X_train.shape[0])
                        X_combined = np.vstack([X_train, generated])
                        y_combined = np.tile(y_train_sparse, 2)
                        classifier.train(X_combined, y_combined)
                        acc = classifier.evaluate(X_test, y_test)['accuracy']
                    elif model_name == 'vae':
                        vae = VAEClassifier()
                        vae.build(input_dim, num_classes, is_binary)
                        vae.train(X_train)
                        acc = vae.evaluate(X_test, y_test)['accuracy']
                    elif model_name == 'dcgan':
                        dcgan = DCGAN(input_dim)
                        dcgan.build_generator()
                        dcgan.build_discriminator()
                        dcgan.build_gan()
                        dcgan.train(X_train)
                        classifier = DCGANClassifier(input_dim)
                        classifier.build(input_dim, num_classes)
                        y_train_sparse = np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train.ravel()
                        generated = dcgan.generate_samples(X_train.shape[0])
                        X_combined = np.vstack([X_train, generated])
                        y_combined = np.tile(y_train_sparse, 2)
                        classifier.train(X_combined, y_combined)
                        acc = classifier.evaluate(X_test, y_test)['accuracy']
                    elif model_name == 'cgan':
                        y_train_sparse = np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train.ravel()
                        num_classes_cgan = len(np.unique(y_train_sparse))
                        cgan = CGAN(input_dim, num_classes_cgan)
                        cgan.build()
                        cgan.train(X_train, y_train_sparse)
                        classifier = CGANClassifier(input_dim)
                        classifier.build(input_dim, num_classes_cgan)
                        unique_classes = np.unique(y_train_sparse)
                        n_per_class = X_train.shape[0] // len(unique_classes)
                        all_generated = []
                        for cls in unique_classes:
                            gen = cgan.generate_samples(n_per_class, cls)
                            all_generated.append(gen)
                        generated = np.vstack(all_generated)
                        y_generated = np.repeat(unique_classes, n_per_class)
                        X_combined = np.vstack([X_train, generated])
                        y_combined = np.hstack([y_train_sparse, y_generated])
                        classifier.train(X_combined, y_combined)
                        y_test_sparse = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test.ravel()
                        acc = classifier.evaluate(X_test, y_test_sparse)['accuracy']
                    elif model_name == 'ctgan':
                        ctgan = CTGANModel()
                        ctgan.build(input_dim, num_classes, is_binary)
                        ctgan.train(X_train)
                        acc = ctgan.evaluate(X_test, y_test)['accuracy']
                    elif model_name == 'diffusion':
                        diffusion = TabDDPM(input_dim)
                        diffusion.build(input_dim, num_classes, is_binary)
                        diffusion.train(X_train)
                        acc = diffusion.evaluate(X_test, y_test)['accuracy']
                    else:
                        st.warning(f'{model_name} not implemented')
                        acc = 0.0
                        
                    train_time = time.time() - model_start
                    results[f'{model_name}_acc'] = max(0, min(1, acc))
                    results[f'{model_name}_time'] = max(0, train_time)
                    st.success(f'{model_name.upper()}: {acc:.3f} acc, {train_time:.1f}s')
                    
                except Exception as e:
                    st.error(f'{model_name} failed: {str(e)[:100]}...')
                    results[f'{model_name}_acc'] = 0.0
                    results[f'{model_name}_time'] = 0.0
                    
            progress_bar.progress((i + 1) / len(selected_models))
        
        st.subheader('Results Table')
        df_results = pd.DataFrame([results])
        st.dataframe(df_results, width='stretch')
        
        st.subheader('Accuracy Chart')
        acc_data = {k.replace('_acc',''): v for k, v in results.items() if '_acc' in k}
        if acc_data:
            st.bar_chart(acc_data, height=400)
        else:
            st.warning('No valid accuracies to plot')
            
        st.subheader('Training Time Chart')
        time_data = {k.replace('_time',''): v for k, v in results.items() if '_time' in k}
        if time_data:
            st.bar_chart(time_data, height=400)
        else:
            st.warning('No valid times to plot')
        
        st.success('All models complete!')
        
        # Save results
        os.makedirs('results', exist_ok=True)
        with open(f'results/{selected_dataset}_gui_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        st.info(f'Saved: results/{selected_dataset}_gui_results.json')

st.info('Run: streamlit run gui.py | Select dataset/models → Run → Charts/Results')

