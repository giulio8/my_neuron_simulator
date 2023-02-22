def shortcut(s):
    x = ""
    y = ["a", "e", "i", "o", "u"]

    for l in s:
        if ((l in y) == False):
            x += l
    return x

parola = shortcut("pisello")
print(parola)