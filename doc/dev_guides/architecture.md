mx.Disposable is core class of most objects.

Here some differences with classic "disposable pattern".

Disposable must be disposed explicitly.
If Disposable was not disposed, then warning occurs in the console.
No implicit disposition at garbage collection.
If object was not disposed, the warning will be printed at the garbage collection.
Use tree style with Disposable to design the app correctly.

When you call object.dispose(), no one should use object at this point.
All object users KNOW when the object is no longer usable,
and finalizes interaction with it.
Disposable should be disposed only once, otherwise Exception raised.

There is no .is_disposed() method, because you must know when object is disposed.

There is .dispose_with() method available.
All attached disposables will be disposed in FILO order.


## Model-ViewController.

Model and ViewController creation and disposition always done by the same parent entity.
Model and ViewController never disposes itself.
If something goes wrong with the model, implement error state of model for user.

ViewController always attaches to existing non disposed Model.
ViewController always detaches(disposes) from existing non disposed Model.

ViewController always knows when sub-Models will be created/disposed inside Model,
therefore sub-ViewController must be disposed before sub-Model disposition.

Mx-controls are Disposables. 
Event, EventConnection are Disposables

You must dispose EventConnection if func operates disposable data before(or with) this data will be disposed.
Example:    signal.listen(lambda: data.process()).dispose_with(data)


## AsyncX task system.

When you design a Task, keep in mind that Task can be cancelled from any yield-interruption point.

Task should implement cancel logic in order to finish correctly.
You may have to dispose outter resources such as files or ax.Thread inside task.

Task can be cancelled from any thread using 
1) direct .cancel()
2) via TaskGroup/Taskset .cancel_all()
3) cancelled by cancel it's parent. Task is detached from parent if attached to TaskGroup (not to TaskSet).


@ax.protected_task cannot be cancelled by it's creator.