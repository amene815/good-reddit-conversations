#!/usr/bin/env python3
import pandas as pd
import pdb
import csv

class Threads:
    def __init__(self):
        self.threads = {}

    def new_thread(self,post):
        self.threads[post[1].link_id] = Thread(post)

    def add(self,post):
        # print(self.threads)
        thread = self.threads[post[1].link_id]
        thread.add(post)

    def threads(self):
        return self.threads

    def exists(self, link_id):
        if link_id in self.threads.keys():
            return True
        else:
            return False

    # def treeify(self):
    #     for key, thread in self.threads:
    #         thread.treeify()

    def calc_max_len(self):
        max_lens = []
        for thread_id in self.threads:
            thread = self.threads[thread_id]
            for id in thread.calc_max_len():
                max_lens.append(id)
        return max_lens


class Thread:
    def __init__(self,post):
        # print(post)
        self.name = post[1].link_id
        self.posts = {}

    def add(self,post):
        self.posts[post[1].id] = Post(post)

    def treeify(self):
        for id in self.posts:
            post = self.posts[id]
            if post.parent == self.name[3:]: # if this happens it is because the post is a reply to the thread posting, as far as I can tell, not because it is the root
                continue

            try:
                parent = self.posts[post.parent]
                parent.children.append(post)
            except:
                post.parent = None

    def update_len(self, post):
        x = -1
        for child in post.children:
            x = x if x > child.max_following_posts else child.max_following_posts

        post.max_following_posts = x + 1

        if post.parent != self.name[3:] and post.parent != None:
            self.update_len(self.posts[post.parent])

    def calc_max_len(self):
        self.treeify()
        for id in self.posts:
            post = self.posts[id]
            if len(post.children) == 0:
                self.update_len(post)

        return [(id,self.posts[id].body,self.posts[id].max_following_posts) for id in self.posts]


class Post:
    def __init__(self,post):
        # try:
        self.id = post[1].id
        self.parent = post[1].parent_id[3:]
        self.body = post[1].body
        self.children = []
        self.max_following_posts = -1
        # except Exception as e:
        #     breakpoint()


def main(args=None):
    frames = [pd.read_csv(x) for x in ["data/2017-11.csv","data/2017-12.csv","data/2018-01.csv","data/2018-02.csv","data/2018-03.csv"]]
    df = pd.concat(frames)
    info_for_sorting = pd.concat([df[[x]] for x in ["id","link_id","parent_id","body"]],axis = 1)
    info_for_sorting = info_for_sorting.dropna(axis=0, how='any',subset=["id","link_id","parent_id","body"])

    threads = Threads()

    for row in info_for_sorting.iterrows():
        # breakpoint()
        # print(row)
        if threads.exists(row[1].link_id):
            threads.add(row)
        else:
            threads.new_thread(row)
            threads.add(row)


    # dic = threads.threads
    # thread = dic["t3_79zwnf"]
    # thread.treeify()
    # max_lens = thread.calc_max_len()

    max_lens = threads.calc_max_len()

    with open('data\\non-tokenized-data.csv', 'w', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(['id', 'body', 'max_len'])
        writer.writerows(max_lens)
        # for row in max_lens:
        #     writer.writerow(row)
    # print(max_lens[0])
    # breakpoint()

if __name__ == "__main__":
    main()
