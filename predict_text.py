import ktrain

predictor=ktrain.load_predictor('toxic_comment')

def check_image_toxic_text(text):
    flag=0
    probs=predictor.predict(text)
    flags = [1 for prob in probs if prob[1] > 0.75]
    for flag in flags:
        if flag:
            return flag
    return flag
