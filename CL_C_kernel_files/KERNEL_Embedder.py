import os





def embed_head(entry_name:str)->str:
    
    head:str="""
#pragma once
#include <string>
class cl_embed {
    public:
    """
    return head


def embed_tail(entry_name:str)->str:
    return entry_name + "\n};"


def get_all_kernels(embed_:str)->str:
        
    for i in os.listdir("./"):
        if i.endswith(".cl"):
            
            clfile=open("./"+i, "r",encoding="utf-8")
            #read_and_embed(clfile, embed_)
            embed_ = read_and_embed(clfile, embed_)
            clfile.close()
    
    return embed_tail(embed_)
    pass


def get_title(text:str)->str:
    void_loc = text.find("void ") + 5
    arg_loc = text.find("(")

    return text[void_loc:arg_loc]

def add_title(title:str, main_text:str)->str:
    return "std::string "+title+" = \n"+main_text + "\t;\n"

def read_and_embed(file, got_str:str)->str:
    main_text:str=""
    title:str = ""
    for i in file.readlines():
        line:str=i[:-1]
        if line.find("//-ne")!=-1 or line.find("//No_Embed")!=-1 or line.find("printf")!=-1:
        
            pass
        else:
            if line.find("__kernel")!=-1:
                title = get_title(line)
            line="\t\""+line+"\\n\""+"\n"
            main_text+=line




    return got_str + add_title(title, main_text)





embed_string:str=""
embed_string = embed_head(embed_string)

out_file=open("./cl_embedded.h","w", encoding="utf-8")
out_file.write(get_all_kernels(embed_string))
out_file.close()



