# output command line arguments line by line
def command_line_parameter(args):
    # calculate the length of key and value to ensure the output table is in normal shape
    key_max_length = 0
    value_max_length = 0
    for key, value in args.__dict__.items():
        key_max_length = len(key) if len(key) > key_max_length else key_max_length
        value_max_length = len(str(value)) if len(str(value)) > value_max_length else value_max_length

    # 8 = icons and spaces on both sides + the space in the middle, average on both sides
    string_length = 8 + 8 + 8 + key_max_length + value_max_length
    table_title = "COMMAND LINE PARAMETERS"
    # If the filling area cannot be evenly divided, handle it according to different situations
    # If the remainder is 1, fill the icon at the end
    add_end_icon = True if (string_length - len(table_title)) % 4 == 1 else False
    # If the remainder is 2, fill the icon at the head and tail
    add_sides_icon = True if (string_length - len(table_title)) % 4 == 2 else False
    # If the remainder is 3, add 1 to each icon and space, and subtract 1 from the last icon.
    delete_end_icon = True if (string_length - len(table_title)) % 4 == 3 else False
    icon_length = space_length = (string_length - len(table_title)) // 4
    icon_length = icon_length + 1 if (add_sides_icon or delete_end_icon) else icon_length
    space_length = space_length + 1 if delete_end_icon else space_length
    front_icon = rear_icon = "*" * icon_length
    front_space = rear_space = " " * space_length
    rear_icon = "*" * (icon_length + 1) if add_end_icon else rear_icon
    rear_icon = "*" * (icon_length - 1) if delete_end_icon else rear_icon
    print("*" * string_length)
    print(front_icon + front_space + table_title + rear_space + rear_icon)
    print("*" * string_length)

    for key, value in args.__dict__.items():
        # 16 = icons and spaces on both sides
        space_total = (string_length - 16 - len(key) - len(str(value)))
        space_before = 3 + key_max_length - len(key)
        space_after = space_total - space_before - 2
        print("*" * 4 + " " * 4 + f"{key}"
              + " " * space_before + "||" + " " * space_after
              + f"{value}" + " " * 4 + "*" * 4)
    print("*" * string_length)
